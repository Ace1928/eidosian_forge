import base64
import binascii
import os
import random
import re
import socket
import time
import warnings
from email.utils import parseaddr
from io import BytesIO
from typing import Type
from zope.interface import implementer
from twisted import cred
from twisted.copyright import longversion
from twisted.internet import defer, error, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.interfaces import ISSLTransport, ITLSTransport
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.python.runtime import platform
import codecs
class ESMTPClient(SMTPClient):
    """
    A client for sending emails over ESMTP.

    @ivar heloFallback: Whether or not to fall back to plain SMTP if the C{EHLO}
        command is not recognised by the server. If L{requireAuthentication} is
        C{True}, or L{requireTransportSecurity} is C{True} and the connection is
        not over TLS, this fallback flag will not be honored.
    @type heloFallback: L{bool}

    @ivar requireAuthentication: If C{True}, refuse to proceed if authentication
        cannot be performed. Overrides L{heloFallback}.
    @type requireAuthentication: L{bool}

    @ivar requireTransportSecurity: If C{True}, refuse to proceed if the
        transport cannot be secured. If the transport layer is not already
        secured via TLS, this will override L{heloFallback}.
    @type requireAuthentication: L{bool}

    @ivar context: The context factory to use for STARTTLS, if desired.
    @type context: L{IOpenSSLClientConnectionCreator}

    @ivar _tlsMode: Whether or not the connection is over TLS.
    @type _tlsMode: L{bool}
    """
    heloFallback = True
    requireAuthentication = False
    requireTransportSecurity = False
    context = None
    _tlsMode = False

    def __init__(self, secret, contextFactory=None, *args, **kw):
        SMTPClient.__init__(self, *args, **kw)
        self.authenticators = []
        self.secret = secret
        self.context = contextFactory

    def __getattr__(self, name):
        if name == 'tlsMode':
            warnings.warn('tlsMode attribute of twisted.mail.smtp.ESMTPClient is deprecated since Twisted 13.0', category=DeprecationWarning, stacklevel=2)
            return self._tlsMode
        else:
            raise AttributeError('%s instance has no attribute %r' % (self.__class__.__name__, name))

    def __setattr__(self, name, value):
        if name == 'tlsMode':
            warnings.warn('tlsMode attribute of twisted.mail.smtp.ESMTPClient is deprecated since Twisted 13.0', category=DeprecationWarning, stacklevel=2)
            self._tlsMode = value
        else:
            self.__dict__[name] = value

    def esmtpEHLORequired(self, code=-1, resp=None):
        """
        Fail because authentication is required, but the server does not support
        ESMTP, which is required for authentication.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}
        """
        self.sendError(EHLORequiredError(502, b'Server does not support ESMTP Authentication', self.log.str()))

    def esmtpAUTHRequired(self, code=-1, resp=None):
        """
        Fail because authentication is required, but the server does not support
        any schemes we support.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}
        """
        tmp = []
        for a in self.authenticators:
            tmp.append(a.getName().upper())
        auth = b'[%s]' % b', '.join(tmp)
        self.sendError(AUTHRequiredError(502, b'Server does not support Client Authentication schemes %s' % auth, self.log.str()))

    def esmtpTLSRequired(self, code=-1, resp=None):
        """
        Fail because TLS is required and the server does not support it.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}
        """
        self.sendError(TLSRequiredError(502, b'Server does not support secure communication via TLS / SSL', self.log.str()))

    def esmtpTLSFailed(self, code=-1, resp=None):
        """
        Fail because the TLS handshake wasn't able to be completed.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}
        """
        self.sendError(TLSError(code, b'Could not complete the SSL/TLS handshake', self.log.str()))

    def esmtpAUTHDeclined(self, code=-1, resp=None):
        """
        Fail because the authentication was rejected.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}
        """
        self.sendError(AUTHDeclinedError(code, resp, self.log.str()))

    def esmtpAUTHMalformedChallenge(self, code=-1, resp=None):
        """
        Fail because the server sent a malformed authentication challenge.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}
        """
        self.sendError(AuthenticationError(501, b'Login failed because the SMTP Server returned a malformed Authentication Challenge', self.log.str()))

    def esmtpAUTHServerError(self, code=-1, resp=None):
        """
        Fail because of some other authentication error.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}
        """
        self.sendError(AuthenticationError(code, resp, self.log.str()))

    def registerAuthenticator(self, auth):
        """
        Registers an Authenticator with the ESMTPClient. The ESMTPClient will
        attempt to login to the SMTP Server in the order the Authenticators are
        registered. The most secure Authentication mechanism should be
        registered first.

        @param auth: The Authentication mechanism to register
        @type auth: L{IClientAuthentication} implementor

        @return: L{None}
        """
        self.authenticators.append(auth)

    def connectionMade(self):
        """
        Called when a connection has been made, and triggers sending an C{EHLO}
        to the server.
        """
        self._tlsMode = ISSLTransport.providedBy(self.transport)
        SMTPClient.connectionMade(self)
        self._okresponse = self.esmtpState_ehlo

    def esmtpState_ehlo(self, code, resp):
        """
        Send an C{EHLO} to the server.

        If L{heloFallback} is C{True}, and there is no requirement for TLS or
        authentication, the client will fall back to basic SMTP.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}

        @return: L{None}
        """
        self._expected = SUCCESS
        self._okresponse = self.esmtpState_serverConfig
        self._failresponse = self.esmtpEHLORequired
        if self._tlsMode:
            needTLS = False
        else:
            needTLS = self.requireTransportSecurity
        if self.heloFallback and (not self.requireAuthentication) and (not needTLS):
            self._failresponse = self.smtpState_helo
        self.sendLine(b'EHLO ' + self.identity)

    def esmtpState_serverConfig(self, code, resp):
        """
        Handle a positive response to the I{EHLO} command by parsing the
        capabilities in the server's response and then taking the most
        appropriate next step towards entering a mail transaction.
        """
        items = {}
        for line in resp.splitlines():
            e = line.split(None, 1)
            if len(e) > 1:
                items[e[0]] = e[1]
            else:
                items[e[0]] = None
        self.tryTLS(code, resp, items)

    def tryTLS(self, code, resp, items):
        """
        Take a necessary step towards being able to begin a mail transaction.

        The step may be to ask the server to being a TLS session.  If TLS is
        already in use or not necessary and not available then the step may be
        to authenticate with the server.  If TLS is necessary and not available,
        fail the mail transmission attempt.

        This is an internal helper method.

        @param code: The server status code from the most recently received
            server message.
        @type code: L{int}

        @param resp: The server status response from the most recently received
            server message.
        @type resp: L{bytes}

        @param items: A mapping of ESMTP extensions offered by the server.  Keys
            are extension identifiers and values are the associated values.
        @type items: L{dict} mapping L{bytes} to L{bytes}

        @return: L{None}
        """
        hasTLS = self._tlsMode
        canTLS = self.context and b'STARTTLS' in items
        mustTLS = self.requireTransportSecurity
        if hasTLS or not (canTLS or mustTLS):
            self.authenticate(code, resp, items)
        elif canTLS:
            self._expected = [220]
            self._okresponse = self.esmtpState_starttls
            self._failresponse = self.esmtpTLSFailed
            self.sendLine(b'STARTTLS')
        else:
            self.esmtpTLSRequired()

    def esmtpState_starttls(self, code, resp):
        """
        Handle a positive response to the I{STARTTLS} command by starting a new
        TLS session on C{self.transport}.

        Upon success, re-handshake with the server to discover what capabilities
        it has when TLS is in use.
        """
        try:
            self.transport.startTLS(self.context)
            self._tlsMode = True
        except BaseException:
            log.err()
            self.esmtpTLSFailed(451)
        self.esmtpState_ehlo(code, resp)

    def authenticate(self, code, resp, items):
        if self.secret and items.get(b'AUTH'):
            schemes = items[b'AUTH'].split()
            tmpSchemes = {}
            for s in schemes:
                tmpSchemes[s.upper()] = 1
            for a in self.authenticators:
                auth = a.getName().upper()
                if auth in tmpSchemes:
                    self._authinfo = a
                    if auth == b'PLAIN':
                        self._okresponse = self.smtpState_from
                        self._failresponse = self._esmtpState_plainAuth
                        self._expected = [235]
                        challenge = base64.b64encode(self._authinfo.challengeResponse(self.secret, 1))
                        self.sendLine(b'AUTH %s %s' % (auth, challenge))
                    else:
                        self._expected = [334]
                        self._okresponse = self.esmtpState_challenge
                        self._failresponse = self.esmtpAUTHServerError
                        self.sendLine(b'AUTH ' + auth)
                    return
        if self.requireAuthentication:
            self.esmtpAUTHRequired()
        else:
            self.smtpState_from(code, resp)

    def _esmtpState_plainAuth(self, code, resp):
        self._okresponse = self.smtpState_from
        self._failresponse = self.esmtpAUTHDeclined
        self._expected = [235]
        challenge = base64.b64encode(self._authinfo.challengeResponse(self.secret, 2))
        self.sendLine(b'AUTH PLAIN ' + challenge)

    def esmtpState_challenge(self, code, resp):
        self._authResponse(self._authinfo, resp)

    def _authResponse(self, auth, challenge):
        self._failresponse = self.esmtpAUTHDeclined
        try:
            challenge = base64.b64decode(challenge)
        except binascii.Error:
            self.sendLine(b'*')
            self._okresponse = self.esmtpAUTHMalformedChallenge
            self._failresponse = self.esmtpAUTHMalformedChallenge
        else:
            resp = auth.challengeResponse(self.secret, challenge)
            self._expected = [235, 334]
            self._okresponse = self.smtpState_maybeAuthenticated
            self.sendLine(base64.b64encode(resp))

    def smtpState_maybeAuthenticated(self, code, resp):
        """
        Called to handle the next message from the server after sending a
        response to a SASL challenge.  The server response might be another
        challenge or it might indicate authentication has succeeded.
        """
        if code == 235:
            del self._authinfo
            self.smtpState_from(code, resp)
        else:
            self._authResponse(self._authinfo, resp)