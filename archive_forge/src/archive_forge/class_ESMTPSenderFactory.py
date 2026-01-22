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
class ESMTPSenderFactory(SMTPSenderFactory):
    """
    Utility factory for sending emails easily.

    @type currentProtocol: L{ESMTPSender}
    @ivar currentProtocol: The current running protocol as made by
        L{buildProtocol}.
    """
    protocol = ESMTPSender

    def __init__(self, username, password, fromEmail, toEmail, file, deferred, retries=5, timeout=None, contextFactory=None, heloFallback=False, requireAuthentication=True, requireTransportSecurity=True, hostname=None):
        SMTPSenderFactory.__init__(self, fromEmail, toEmail, file, deferred, retries, timeout)
        self.username = username
        self.password = password
        self._contextFactory = contextFactory
        self._heloFallback = heloFallback
        self._requireAuthentication = requireAuthentication
        self._requireTransportSecurity = requireTransportSecurity
        self._hostname = hostname

    def buildProtocol(self, addr):
        """
        Build an L{ESMTPSender} protocol configured with C{heloFallback},
        C{requireAuthentication}, and C{requireTransportSecurity} as specified
        in L{__init__}.

        This sets L{currentProtocol} on the factory, as well as returning it.

        @rtype: L{ESMTPSender}
        """
        p = self.protocol(self.username, self.password, self._contextFactory, self.domain, self.nEmails * 2 + 2, hostname=self._hostname)
        p.heloFallback = self._heloFallback
        p.requireAuthentication = self._requireAuthentication
        p.requireTransportSecurity = self._requireTransportSecurity
        p.factory = self
        p.timeout = self.timeout
        self.currentProtocol = p
        self.result.addBoth(self._removeProtocol)
        return p