import re
import struct
import zlib
from base64 import b64encode
from datetime import datetime, timedelta
from hashlib import md5
from webob.byterange import ContentRange
from webob.cachecontrol import CacheControl, serialize_cache_control
from webob.compat import (
from webob.cookies import Cookie, make_cookie
from webob.datetime_utils import (
from webob.descriptors import (
from webob.headers import ResponseHeaders
from webob.request import BaseRequest
from webob.util import status_generic_reasons, status_reasons, warn_deprecation
class ResponseBodyFile(object):
    mode = 'wb'
    closed = False

    def __init__(self, response):
        """
        Represents a :class:`~Response` as a file like object.
        """
        self.response = response
        self.write = response.write

    def __repr__(self):
        return '<body_file for %r>' % self.response
    encoding = property(lambda self: self.response.charset, doc='The encoding of the file (inherited from response.charset)')

    def writelines(self, seq):
        """
        Write a sequence of lines to the response.
        """
        for item in seq:
            self.write(item)

    def close(self):
        raise NotImplementedError('Response bodies cannot be closed')

    def flush(self):
        pass

    def tell(self):
        """
        Provide the current location where we are going to start writing.
        """
        if not self.response.has_body:
            return 0
        return sum([len(chunk) for chunk in self.response.app_iter])