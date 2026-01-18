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
def unset_cookie(self, name, strict=True):
    """
        Unset a cookie with the given name (remove it from the response).
        """
    existing = self.headers.getall('Set-Cookie')
    if not existing and (not strict):
        return
    cookies = Cookie()
    for header in existing:
        cookies.load(header)
    if isinstance(name, text_type):
        name = name.encode('utf8')
    if name in cookies:
        del cookies[name]
        del self.headers['Set-Cookie']
        for m in cookies.values():
            self.headerlist.append(('Set-Cookie', m.serialize()))
    elif strict:
        raise KeyError('No cookie has been set with the name %r' % name)