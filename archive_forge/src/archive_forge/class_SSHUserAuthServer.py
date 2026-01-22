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
class SSHUserAuthServer(service.SSHService):
    """
    A service implementing the server side of the 'ssh-userauth' service.  It
    is used to authenticate the user on the other side as being able to access
    this server.

    @ivar name: the name of this service: 'ssh-userauth'
    @type name: L{bytes}
    @ivar authenticatedWith: a list of authentication methods that have
        already been used.
    @type authenticatedWith: L{list}
    @ivar loginTimeout: the number of seconds we wait before disconnecting
        the user for taking too long to authenticate
    @type loginTimeout: L{int}
    @ivar attemptsBeforeDisconnect: the number of failed login attempts we
        allow before disconnecting.
    @type attemptsBeforeDisconnect: L{int}
    @ivar loginAttempts: the number of login attempts that have been made
    @type loginAttempts: L{int}
    @ivar passwordDelay: the number of seconds to delay when the user gives
        an incorrect password
    @type passwordDelay: L{int}
    @ivar interfaceToMethod: a L{dict} mapping credential interfaces to
        authentication methods.  The server checks to see which of the
        cred interfaces have checkers and tells the client that those methods
        are valid for authentication.
    @type interfaceToMethod: L{dict}
    @ivar supportedAuthentications: A list of the supported authentication
        methods.
    @type supportedAuthentications: L{list} of L{bytes}
    @ivar user: the last username the client tried to authenticate with
    @type user: L{bytes}
    @ivar method: the current authentication method
    @type method: L{bytes}
    @ivar nextService: the service the user wants started after authentication
        has been completed.
    @type nextService: L{bytes}
    @ivar portal: the L{twisted.cred.portal.Portal} we are using for
        authentication
    @type portal: L{twisted.cred.portal.Portal}
    @ivar clock: an object with a callLater method.  Stubbed out for testing.
    """
    name = b'ssh-userauth'
    loginTimeout = 10 * 60 * 60
    attemptsBeforeDisconnect = 20
    passwordDelay = 1
    clock = reactor
    interfaceToMethod = {credentials.ISSHPrivateKey: b'publickey', credentials.IUsernamePassword: b'password'}
    _log = Logger()

    def serviceStarted(self):
        """
        Called when the userauth service is started.  Set up instance
        variables, check if we should allow password authentication (only
        allow if the outgoing connection is encrypted) and set up a login
        timeout.
        """
        self.authenticatedWith = []
        self.loginAttempts = 0
        self.user = None
        self.nextService = None
        self.portal = self.transport.factory.portal
        self.supportedAuthentications = []
        for i in self.portal.listCredentialsInterfaces():
            if i in self.interfaceToMethod:
                self.supportedAuthentications.append(self.interfaceToMethod[i])
        if not self.transport.isEncrypted('in'):
            if b'password' in self.supportedAuthentications:
                self.supportedAuthentications.remove(b'password')
        self._cancelLoginTimeout = self.clock.callLater(self.loginTimeout, self.timeoutAuthentication)

    def serviceStopped(self):
        """
        Called when the userauth service is stopped.  Cancel the login timeout
        if it's still going.
        """
        if self._cancelLoginTimeout:
            self._cancelLoginTimeout.cancel()
            self._cancelLoginTimeout = None

    def timeoutAuthentication(self):
        """
        Called when the user has timed out on authentication.  Disconnect
        with a DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE message.
        """
        self._cancelLoginTimeout = None
        self.transport.sendDisconnect(transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE, b'you took too long')

    def tryAuth(self, kind, user, data):
        """
        Try to authenticate the user with the given method.  Dispatches to a
        auth_* method.

        @param kind: the authentication method to try.
        @type kind: L{bytes}
        @param user: the username the client is authenticating with.
        @type user: L{bytes}
        @param data: authentication specific data sent by the client.
        @type data: L{bytes}
        @return: A Deferred called back if the method succeeded, or erred back
            if it failed.
        @rtype: C{defer.Deferred}
        """
        self._log.debug('{user!r} trying auth {kind!r}', user=user, kind=kind)
        if kind not in self.supportedAuthentications:
            return defer.fail(error.ConchError('unsupported authentication, failing'))
        kind = nativeString(kind.replace(b'-', b'_'))
        f = getattr(self, f'auth_{kind}', None)
        if f:
            ret = f(data)
            if not ret:
                return defer.fail(error.ConchError(f'{kind} return None instead of a Deferred'))
            else:
                return ret
        return defer.fail(error.ConchError(f'bad auth type: {kind}'))

    def ssh_USERAUTH_REQUEST(self, packet):
        """
        The client has requested authentication.  Payload::
            string user
            string next service
            string method
            <authentication specific data>

        @type packet: L{bytes}
        """
        user, nextService, method, rest = getNS(packet, 3)
        if user != self.user or nextService != self.nextService:
            self.authenticatedWith = []
        self.user = user
        self.nextService = nextService
        self.method = method
        d = self.tryAuth(method, user, rest)
        if not d:
            self._ebBadAuth(failure.Failure(error.ConchError('auth returned none')))
            return
        d.addCallback(self._cbFinishedAuth)
        d.addErrback(self._ebMaybeBadAuth)
        d.addErrback(self._ebBadAuth)
        return d

    def _cbFinishedAuth(self, result):
        """
        The callback when user has successfully been authenticated.  For a
        description of the arguments, see L{twisted.cred.portal.Portal.login}.
        We start the service requested by the user.
        """
        interface, avatar, logout = result
        self.transport.avatar = avatar
        self.transport.logoutFunction = logout
        service = self.transport.factory.getService(self.transport, self.nextService)
        if not service:
            raise error.ConchError(f'could not get next service: {self.nextService}')
        self._log.debug('{user!r} authenticated with {method!r}', user=self.user, method=self.method)
        self.transport.sendPacket(MSG_USERAUTH_SUCCESS, b'')
        self.transport.setService(service())

    def _ebMaybeBadAuth(self, reason):
        """
        An intermediate errback.  If the reason is
        error.NotEnoughAuthentication, we send a MSG_USERAUTH_FAILURE, but
        with the partial success indicator set.

        @type reason: L{twisted.python.failure.Failure}
        """
        reason.trap(error.NotEnoughAuthentication)
        self.transport.sendPacket(MSG_USERAUTH_FAILURE, NS(b','.join(self.supportedAuthentications)) + b'\xff')

    def _ebBadAuth(self, reason):
        """
        The final errback in the authentication chain.  If the reason is
        error.IgnoreAuthentication, we simply return; the authentication
        method has sent its own response.  Otherwise, send a failure message
        and (if the method is not 'none') increment the number of login
        attempts.

        @type reason: L{twisted.python.failure.Failure}
        """
        if reason.check(error.IgnoreAuthentication):
            return
        if self.method != b'none':
            self._log.debug('{user!r} failed auth {method!r}', user=self.user, method=self.method)
            if reason.check(UnauthorizedLogin):
                self._log.debug('unauthorized login: {message}', message=reason.getErrorMessage())
            elif reason.check(error.ConchError):
                self._log.debug('reason: {reason}', reason=reason.getErrorMessage())
            else:
                self._log.failure('Error checking auth for user {user}', failure=reason, user=self.user)
            self.loginAttempts += 1
            if self.loginAttempts > self.attemptsBeforeDisconnect:
                self.transport.sendDisconnect(transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE, b'too many bad auths')
                return
        self.transport.sendPacket(MSG_USERAUTH_FAILURE, NS(b','.join(self.supportedAuthentications)) + b'\x00')

    def auth_publickey(self, packet):
        """
        Public key authentication.  Payload::
            byte has signature
            string algorithm name
            string key blob
            [string signature] (if has signature is True)

        Create a SSHPublicKey credential and verify it using our portal.
        """
        hasSig = ord(packet[0:1])
        algName, blob, rest = getNS(packet[1:], 2)
        try:
            keys.Key.fromString(blob)
        except keys.BadKeyError:
            error = 'Unsupported key type {} or bad key'.format(algName.decode('ascii'))
            self._log.error(error)
            return defer.fail(UnauthorizedLogin(error))
        signature = hasSig and getNS(rest)[0] or None
        if hasSig:
            b = NS(self.transport.sessionID) + bytes((MSG_USERAUTH_REQUEST,)) + NS(self.user) + NS(self.nextService) + NS(b'publickey') + bytes((hasSig,)) + NS(algName) + NS(blob)
            c = credentials.SSHPrivateKey(self.user, algName, blob, b, signature)
            return self.portal.login(c, None, interfaces.IConchUser)
        else:
            c = credentials.SSHPrivateKey(self.user, algName, blob, None, None)
            return self.portal.login(c, None, interfaces.IConchUser).addErrback(self._ebCheckKey, packet[1:])

    def _ebCheckKey(self, reason, packet):
        """
        Called back if the user did not sent a signature.  If reason is
        error.ValidPublicKey then this key is valid for the user to
        authenticate with.  Send MSG_USERAUTH_PK_OK.
        """
        reason.trap(error.ValidPublicKey)
        self.transport.sendPacket(MSG_USERAUTH_PK_OK, packet)
        return failure.Failure(error.IgnoreAuthentication())

    def auth_password(self, packet):
        """
        Password authentication.  Payload::
            string password

        Make a UsernamePassword credential and verify it with our portal.
        """
        password = getNS(packet[1:])[0]
        c = credentials.UsernamePassword(self.user, password)
        return self.portal.login(c, None, interfaces.IConchUser).addErrback(self._ebPassword)

    def _ebPassword(self, f):
        """
        If the password is invalid, wait before sending the failure in order
        to delay brute-force password guessing.
        """
        d = defer.Deferred()
        self.clock.callLater(self.passwordDelay, d.callback, f)
        return d