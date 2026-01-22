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
def _store(self, messages, cmd, silent, flags, uid):
    messages = str(messages).encode('ascii')
    encodedFlags = [networkString(flag) for flag in flags]
    if silent:
        cmd = cmd + b'.SILENT'
    store = uid and b'UID STORE' or b'STORE'
    args = b' '.join((messages, cmd, b'(' + b' '.join(encodedFlags) + b')'))
    d = self.sendCommand(Command(store, args, wantResponse=(b'FETCH',)))
    expected = ()
    if not silent:
        expected = ('FLAGS',)
    d.addCallback(self._cbFetch, expected, True)
    return d