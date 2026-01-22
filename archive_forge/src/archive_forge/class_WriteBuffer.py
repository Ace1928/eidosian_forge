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
class WriteBuffer:
    """
    Buffer up a bunch of writes before sending them all to a transport at once.
    """

    def __init__(self, transport, size=8192):
        self.bufferSize = size
        self.transport = transport
        self._length = 0
        self._writes = []

    def write(self, s):
        self._length += len(s)
        self._writes.append(s)
        if self._length > self.bufferSize:
            self.flush()

    def flush(self):
        if self._writes:
            self.transport.writeSequence(self._writes)
            self._writes = []
            self._length = 0