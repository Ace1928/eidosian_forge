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
def _body__set(self, value=b''):
    if not isinstance(value, bytes):
        if isinstance(value, text_type):
            msg = 'You cannot set Response.body to a text object (use Response.text)'
        else:
            msg = 'You can only set the body to a binary type (not %s)' % type(value)
        raise TypeError(msg)
    if self._app_iter is not None:
        self.content_md5 = None
    self._app_iter = [value]
    self.content_length = len(value)