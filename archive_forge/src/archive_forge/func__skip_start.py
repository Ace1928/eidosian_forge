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
def _skip_start(self):
    start, stop = (self.start, self.stop)
    for chunk in self.app_iter:
        self._pos += len(chunk)
        if self._pos < start:
            continue
        elif self._pos == start:
            return b''
        else:
            chunk = chunk[start - self._pos:]
            if stop is not None and self._pos > stop:
                chunk = chunk[:stop - self._pos]
                assert len(chunk) == stop - start
            return chunk
    else:
        raise StopIteration()