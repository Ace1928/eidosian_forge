import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def __cbAuthenticate(self, caps, secret):
    auths = caps.get(b'AUTH', ())
    for scheme in auths:
        if scheme.upper() in self.authenticators:
            cmd = Command(b'AUTHENTICATE', scheme, (), self.__cbContinueAuth, scheme, secret)
            return self.sendCommand(cmd)
    if self.startedTLS:
        return defer.fail(NoSupportedAuthentication(auths, self.authenticators.keys()))
    else:

        def ebStartTLS(err):
            err.trap(IMAP4Exception)
            return defer.fail(NoSupportedAuthentication(auths, self.authenticators.keys()))
        d = self.startTLS()
        d.addErrback(ebStartTLS)
        d.addCallback(lambda _: self.getCapabilities())
        d.addCallback(self.__cbAuthTLS, secret)
        return d