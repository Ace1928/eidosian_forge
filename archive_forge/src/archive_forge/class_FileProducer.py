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
class FileProducer:
    CHUNK_SIZE = 2 ** 2 ** 2 ** 2
    firstWrite = True

    def __init__(self, f):
        self.f = f

    def beginProducing(self, consumer):
        self.consumer = consumer
        self.produce = consumer.write
        d = self._onDone = defer.Deferred()
        self.consumer.registerProducer(self, False)
        return d

    def resumeProducing(self):
        b = b''
        if self.firstWrite:
            b = b'{%d}\r\n' % (self._size(),)
            self.firstWrite = False
        if not self.f:
            return
        b = b + self.f.read(self.CHUNK_SIZE)
        if not b:
            self.consumer.unregisterProducer()
            self._onDone.callback(self)
            self._onDone = self.f = self.consumer = None
        else:
            self.produce(b)

    def pauseProducing(self):
        """
        Pause the producer.  This does nothing.
        """

    def stopProducing(self):
        """
        Stop the producer.  This does nothing.
        """

    def _size(self):
        b = self.f.tell()
        self.f.seek(0, 2)
        e = self.f.tell()
        self.f.seek(b, 0)
        return e - b