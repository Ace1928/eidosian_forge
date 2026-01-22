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
def __cbLogin(self, result, tag):
    iface, avatar, logout = result
    if iface is not IAccount:
        self.sendBadResponse(tag, b'Server error: login returned unexpected value')
        log.err(f'__cbLogin called with {iface!r}, IAccount expected')
    else:
        self.account = avatar
        self._onLogout = logout
        self.sendPositiveResponse(tag, b'LOGIN succeeded')
        self.state = 'auth'
        self.setTimeout(self.POSTAUTH_TIMEOUT)