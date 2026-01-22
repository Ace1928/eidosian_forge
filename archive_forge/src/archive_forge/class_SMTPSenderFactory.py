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
class SMTPSenderFactory(protocol.ClientFactory):
    """
    Utility factory for sending emails easily.

    @type currentProtocol: L{SMTPSender}
    @ivar currentProtocol: The current running protocol returned by
        L{buildProtocol}.

    @type sendFinished: C{bool}
    @ivar sendFinished: When the value is set to True, it means the message has
        been sent or there has been an unrecoverable error or the sending has
        been cancelled. The default value is False.
    """
    domain = DNSNAME
    protocol: Type[SMTPClient] = SMTPSender

    def __init__(self, fromEmail, toEmail, file, deferred, retries=5, timeout=None):
        """
        @param fromEmail: The RFC 2821 address from which to send this
        message.

        @param toEmail: A sequence of RFC 2821 addresses to which to
        send this message.

        @param file: A file-like object containing the message to send.

        @param deferred: A Deferred to callback or errback when sending
        of this message completes.
        @type deferred: L{defer.Deferred}

        @param retries: The number of times to retry delivery of this
        message.

        @param timeout: Period, in seconds, for which to wait for
        server responses, or None to wait forever.
        """
        assert isinstance(retries, int)
        if isinstance(toEmail, str):
            toEmail = [toEmail.encode('ascii')]
        elif isinstance(toEmail, bytes):
            toEmail = [toEmail]
        else:
            toEmailFinal = []
            for _email in toEmail:
                if not isinstance(_email, bytes):
                    _email = _email.encode('ascii')
                toEmailFinal.append(_email)
            toEmail = toEmailFinal
        self.fromEmail = Address(fromEmail)
        self.nEmails = len(toEmail)
        self.toEmail = toEmail
        self.file = file
        self.result = deferred
        self.result.addBoth(self._removeDeferred)
        self.sendFinished = False
        self.currentProtocol = None
        self.retries = -retries
        self.timeout = timeout

    def _removeDeferred(self, result):
        del self.result
        return result

    def clientConnectionFailed(self, connector, err):
        self._processConnectionError(connector, err)

    def clientConnectionLost(self, connector, err):
        self._processConnectionError(connector, err)

    def _processConnectionError(self, connector, err):
        self.currentProtocol = None
        if self.retries < 0 and (not self.sendFinished):
            log.msg('SMTP Client retrying server. Retry: %s' % -self.retries)
            self.file.seek(0, 0)
            connector.connect()
            self.retries += 1
        elif not self.sendFinished:
            if err.check(error.ConnectionDone):
                err.value = SMTPConnectError(-1, 'Unable to connect to server.')
            self.result.errback(err.value)

    def buildProtocol(self, addr):
        p = self.protocol(self.domain, self.nEmails * 2 + 2)
        p.factory = self
        p.timeout = self.timeout
        self.currentProtocol = p
        self.result.addBoth(self._removeProtocol)
        return p

    def _removeProtocol(self, result):
        """
        Remove the protocol created in C{buildProtocol}.

        @param result: The result/error passed to the callback/errback of
            L{defer.Deferred}.

        @return: The C{result} untouched.
        """
        if self.currentProtocol:
            self.currentProtocol = None
        return result