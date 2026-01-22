import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString
class SSHUserAuthClient(service.SSHService):
    """
    A service implementing the client side of 'ssh-userauth'.

    This service will try all authentication methods provided by the server,
    making callbacks for more information when necessary.

    @ivar name: the name of this service: 'ssh-userauth'
    @type name: L{str}
    @ivar preferredOrder: a list of authentication methods that should be used
        first, in order of preference, if supported by the server
    @type preferredOrder: L{list}
    @ivar user: the name of the user to authenticate as
    @type user: L{bytes}
    @ivar instance: the service to start after authentication has finished
    @type instance: L{service.SSHService}
    @ivar authenticatedWith: a list of strings of authentication methods we've tried
    @type authenticatedWith: L{list} of L{bytes}
    @ivar triedPublicKeys: a list of public key objects that we've tried to
        authenticate with
    @type triedPublicKeys: L{list} of L{Key}
    @ivar lastPublicKey: the last public key object we've tried to authenticate
        with
    @type lastPublicKey: L{Key}
    """
    name = b'ssh-userauth'
    preferredOrder = [b'publickey', b'password', b'keyboard-interactive']

    def __init__(self, user, instance):
        self.user = user
        self.instance = instance

    def serviceStarted(self):
        self.authenticatedWith = []
        self.triedPublicKeys = []
        self.lastPublicKey = None
        self.askForAuth(b'none', b'')

    def askForAuth(self, kind, extraData):
        """
        Send a MSG_USERAUTH_REQUEST.

        @param kind: the authentication method to try.
        @type kind: L{bytes}
        @param extraData: method-specific data to go in the packet
        @type extraData: L{bytes}
        """
        self.lastAuth = kind
        self.transport.sendPacket(MSG_USERAUTH_REQUEST, NS(self.user) + NS(self.instance.name) + NS(kind) + extraData)

    def tryAuth(self, kind):
        """
        Dispatch to an authentication method.

        @param kind: the authentication method
        @type kind: L{bytes}
        """
        kind = nativeString(kind.replace(b'-', b'_'))
        self._log.debug('trying to auth with {kind}', kind=kind)
        f = getattr(self, 'auth_' + kind, None)
        if f:
            return f()

    def _ebAuth(self, ignored, *args):
        """
        Generic callback for a failed authentication attempt.  Respond by
        asking for the list of accepted methods (the 'none' method)
        """
        self.askForAuth(b'none', b'')

    def ssh_USERAUTH_SUCCESS(self, packet):
        """
        We received a MSG_USERAUTH_SUCCESS.  The server has accepted our
        authentication, so start the next service.
        """
        self.transport.setService(self.instance)

    def ssh_USERAUTH_FAILURE(self, packet):
        """
        We received a MSG_USERAUTH_FAILURE.  Payload::
            string methods
            byte partial success

        If partial success is C{True}, then the previous method succeeded but is
        not sufficient for authentication. C{methods} is a comma-separated list
        of accepted authentication methods.

        We sort the list of methods by their position in C{self.preferredOrder},
        removing methods that have already succeeded. We then call
        C{self.tryAuth} with the most preferred method.

        @param packet: the C{MSG_USERAUTH_FAILURE} payload.
        @type packet: L{bytes}

        @return: a L{defer.Deferred} that will be callbacked with L{None} as
            soon as all authentication methods have been tried, or L{None} if no
            more authentication methods are available.
        @rtype: C{defer.Deferred} or L{None}
        """
        canContinue, partial = getNS(packet)
        partial = ord(partial)
        if partial:
            self.authenticatedWith.append(self.lastAuth)

        def orderByPreference(meth):
            """
            Invoked once per authentication method in order to extract a
            comparison key which is then used for sorting.

            @param meth: the authentication method.
            @type meth: L{bytes}

            @return: the comparison key for C{meth}.
            @rtype: L{int}
            """
            if meth in self.preferredOrder:
                return self.preferredOrder.index(meth)
            else:
                return len(self.preferredOrder)
        canContinue = sorted((meth for meth in canContinue.split(b',') if meth not in self.authenticatedWith), key=orderByPreference)
        self._log.debug('can continue with: {methods}', methods=canContinue)
        return self._cbUserauthFailure(None, iter(canContinue))

    def _cbUserauthFailure(self, result, iterator):
        if result:
            return
        try:
            method = next(iterator)
        except StopIteration:
            self.transport.sendDisconnect(transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE, b'no more authentication methods available')
        else:
            d = defer.maybeDeferred(self.tryAuth, method)
            d.addCallback(self._cbUserauthFailure, iterator)
            return d

    def ssh_USERAUTH_PK_OK(self, packet):
        """
        This message (number 60) can mean several different messages depending
        on the current authentication type.  We dispatch to individual methods
        in order to handle this request.
        """
        func = getattr(self, 'ssh_USERAUTH_PK_OK_%s' % nativeString(self.lastAuth.replace(b'-', b'_')), None)
        if func is not None:
            return func(packet)
        else:
            self.askForAuth(b'none', b'')

    def ssh_USERAUTH_PK_OK_publickey(self, packet):
        """
        This is MSG_USERAUTH_PK.  Our public key is valid, so we create a
        signature and try to authenticate with it.
        """
        publicKey = self.lastPublicKey
        b = NS(self.transport.sessionID) + bytes((MSG_USERAUTH_REQUEST,)) + NS(self.user) + NS(self.instance.name) + NS(b'publickey') + b'\x01' + NS(publicKey.sshType()) + NS(publicKey.blob())
        d = self.signData(publicKey, b)
        if not d:
            self.askForAuth(b'none', b'')
            return
        d.addCallback(self._cbSignedData)
        d.addErrback(self._ebAuth)

    def ssh_USERAUTH_PK_OK_password(self, packet):
        """
        This is MSG_USERAUTH_PASSWD_CHANGEREQ.  The password given has expired.
        We ask for an old password and a new password, then send both back to
        the server.
        """
        prompt, language, rest = getNS(packet, 2)
        self._oldPass = self._newPass = None
        d = self.getPassword(b'Old Password: ')
        d = d.addCallbacks(self._setOldPass, self._ebAuth)
        d.addCallback(lambda ignored: self.getPassword(prompt))
        d.addCallbacks(self._setNewPass, self._ebAuth)

    def ssh_USERAUTH_PK_OK_keyboard_interactive(self, packet):
        """
        This is MSG_USERAUTH_INFO_RESPONSE.  The server has sent us the
        questions it wants us to answer, so we ask the user and sent the
        responses.
        """
        name, instruction, lang, data = getNS(packet, 3)
        numPrompts = struct.unpack('!L', data[:4])[0]
        data = data[4:]
        prompts = []
        for i in range(numPrompts):
            prompt, data = getNS(data)
            echo = bool(ord(data[0:1]))
            data = data[1:]
            prompts.append((prompt, echo))
        d = self.getGenericAnswers(name, instruction, prompts)
        d.addCallback(self._cbGenericAnswers)
        d.addErrback(self._ebAuth)

    def _cbSignedData(self, signedData):
        """
        Called back out of self.signData with the signed data.  Send the
        authentication request with the signature.

        @param signedData: the data signed by the user's private key.
        @type signedData: L{bytes}
        """
        publicKey = self.lastPublicKey
        self.askForAuth(b'publickey', b'\x01' + NS(publicKey.sshType()) + NS(publicKey.blob()) + NS(signedData))

    def _setOldPass(self, op):
        """
        Called back when we are choosing a new password.  Simply store the old
        password for now.

        @param op: the old password as entered by the user
        @type op: L{bytes}
        """
        self._oldPass = op

    def _setNewPass(self, np):
        """
        Called back when we are choosing a new password.  Get the old password
        and send the authentication message with both.

        @param np: the new password as entered by the user
        @type np: L{bytes}
        """
        op = self._oldPass
        self._oldPass = None
        self.askForAuth(b'password', b'\xff' + NS(op) + NS(np))

    def _cbGenericAnswers(self, responses):
        """
        Called back when we are finished answering keyboard-interactive
        questions.  Send the info back to the server in a
        MSG_USERAUTH_INFO_RESPONSE.

        @param responses: a list of L{bytes} responses
        @type responses: L{list}
        """
        data = struct.pack('!L', len(responses))
        for r in responses:
            data += NS(r.encode('UTF8'))
        self.transport.sendPacket(MSG_USERAUTH_INFO_RESPONSE, data)

    def auth_publickey(self):
        """
        Try to authenticate with a public key.  Ask the user for a public key;
        if the user has one, send the request to the server and return True.
        Otherwise, return False.

        @rtype: L{bool}
        """
        d = defer.maybeDeferred(self.getPublicKey)
        d.addBoth(self._cbGetPublicKey)
        return d

    def _cbGetPublicKey(self, publicKey):
        if not isinstance(publicKey, keys.Key):
            publicKey = None
        if publicKey is not None:
            self.lastPublicKey = publicKey
            self.triedPublicKeys.append(publicKey)
            self._log.debug('using key of type {keyType}', keyType=publicKey.type())
            self.askForAuth(b'publickey', b'\x00' + NS(publicKey.sshType()) + NS(publicKey.blob()))
            return True
        else:
            return False

    def auth_password(self):
        """
        Try to authenticate with a password.  Ask the user for a password.
        If the user will return a password, return True.  Otherwise, return
        False.

        @rtype: L{bool}
        """
        d = self.getPassword()
        if d:
            d.addCallbacks(self._cbPassword, self._ebAuth)
            return True
        else:
            return False

    def auth_keyboard_interactive(self):
        """
        Try to authenticate with keyboard-interactive authentication.  Send
        the request to the server and return True.

        @rtype: L{bool}
        """
        self._log.debug('authing with keyboard-interactive')
        self.askForAuth(b'keyboard-interactive', NS(b'') + NS(b''))
        return True

    def _cbPassword(self, password):
        """
        Called back when the user gives a password.  Send the request to the
        server.

        @param password: the password the user entered
        @type password: L{bytes}
        """
        self.askForAuth(b'password', b'\x00' + NS(password))

    def signData(self, publicKey, signData):
        """
        Sign the given data with the given public key.

        By default, this will call getPrivateKey to get the private key,
        then sign the data using Key.sign().

        This method is factored out so that it can be overridden to use
        alternate methods, such as a key agent.

        @param publicKey: The public key object returned from L{getPublicKey}
        @type publicKey: L{keys.Key}

        @param signData: the data to be signed by the private key.
        @type signData: L{bytes}
        @return: a Deferred that's called back with the signature
        @rtype: L{defer.Deferred}
        """
        key = self.getPrivateKey()
        if not key:
            return
        return key.addCallback(self._cbSignData, signData)

    def _cbSignData(self, privateKey, signData):
        """
        Called back when the private key is returned.  Sign the data and
        return the signature.

        @param privateKey: the private key object
        @type privateKey: L{keys.Key}
        @param signData: the data to be signed by the private key.
        @type signData: L{bytes}
        @return: the signature
        @rtype: L{bytes}
        """
        return privateKey.sign(signData)

    def getPublicKey(self):
        """
        Return a public key for the user.  If no more public keys are
        available, return L{None}.

        This implementation always returns L{None}.  Override it in a
        subclass to actually find and return a public key object.

        @rtype: L{Key} or L{None}
        """
        return None

    def getPrivateKey(self):
        """
        Return a L{Deferred} that will be called back with the private key
        object corresponding to the last public key from getPublicKey().
        If the private key is not available, errback on the Deferred.

        @rtype: L{Deferred} called back with L{Key}
        """
        return defer.fail(NotImplementedError())

    def getPassword(self, prompt=None):
        """
        Return a L{Deferred} that will be called back with a password.
        prompt is a string to display for the password, or None for a generic
        'user@hostname's password: '.

        @type prompt: L{bytes}/L{None}
        @rtype: L{defer.Deferred}
        """
        return defer.fail(NotImplementedError())

    def getGenericAnswers(self, name, instruction, prompts):
        """
        Returns a L{Deferred} with the responses to the promopts.

        @param name: The name of the authentication currently in progress.
        @param instruction: Describes what the authentication wants.
        @param prompts: A list of (prompt, echo) pairs, where prompt is a
        string to display and echo is a boolean indicating whether the
        user's response should be echoed as they type it.
        """
        return defer.fail(NotImplementedError())