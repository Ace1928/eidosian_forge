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