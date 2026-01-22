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
class MessageProducer:
    CHUNK_SIZE = 2 ** 2 ** 2 ** 2
    _uuid4 = staticmethod(uuid.uuid4)

    def __init__(self, msg, buffer=None, scheduler=None):
        """
        Produce this message.

        @param msg: The message I am to produce.
        @type msg: L{IMessage}

        @param buffer: A buffer to hold the message in.  If None, I will
            use a L{tempfile.TemporaryFile}.
        @type buffer: file-like
        """
        self.msg = msg
        if buffer is None:
            buffer = tempfile.TemporaryFile()
        self.buffer = buffer
        if scheduler is None:
            scheduler = iterateInReactor
        self.scheduler = scheduler
        self.write = self.buffer.write

    def beginProducing(self, consumer):
        self.consumer = consumer
        return self.scheduler(self._produce())

    def _produce(self):
        headers = self.msg.getHeaders(True)
        boundary = None
        if self.msg.isMultipart():
            content = headers.get('content-type')
            parts = [x.split('=', 1) for x in content.split(';')[1:]]
            parts = {k.lower().strip(): v for k, v in parts}
            boundary = parts.get('boundary')
            if boundary is None:
                boundary = f'----={self._uuid4().hex}'
                headers['content-type'] += f'; boundary="{boundary}"'
            elif boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
            boundary = networkString(boundary)
        self.write(_formatHeaders(headers))
        self.write(b'\r\n')
        if self.msg.isMultipart():
            for p in subparts(self.msg):
                self.write(b'\r\n--' + boundary + b'\r\n')
                yield MessageProducer(p, self.buffer, self.scheduler).beginProducing(None)
            self.write(b'\r\n--' + boundary + b'--\r\n')
        else:
            f = self.msg.getBodyFile()
            while True:
                b = f.read(self.CHUNK_SIZE)
                if b:
                    self.buffer.write(b)
                    yield None
                else:
                    break
        if self.consumer:
            self.buffer.seek(0, 0)
            yield FileProducer(self.buffer).beginProducing(self.consumer).addCallback(lambda _: self)