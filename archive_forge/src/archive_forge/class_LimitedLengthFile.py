import binascii
import io
import os
import re
import sys
import tempfile
import mimetypes
import warnings
from webob.acceptparse import (
from webob.cachecontrol import (
from webob.compat import (
from webob.cookies import RequestCookies
from webob.descriptors import (
from webob.etag import (
from webob.headers import EnvironHeaders
from webob.multidict import (
class LimitedLengthFile(io.RawIOBase):

    def __init__(self, file, maxlen):
        self.file = file
        self.maxlen = maxlen
        self.remaining = maxlen

    def __repr__(self):
        return '<%s(%r, maxlen=%s)>' % (self.__class__.__name__, self.file, self.maxlen)

    def fileno(self):
        return self.file.fileno()

    @staticmethod
    def readable():
        return True

    def readinto(self, buff):
        if not self.remaining:
            return 0
        sz0 = min(len(buff), self.remaining)
        data = self.file.read(sz0)
        sz = len(data)
        self.remaining -= sz
        if sz < sz0 and self.remaining:
            raise DisconnectionError('The client disconnected while sending the body (%d more bytes were expected)' % (self.remaining,))
        buff[:sz] = data
        return sz