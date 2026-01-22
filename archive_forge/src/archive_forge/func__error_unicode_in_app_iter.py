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
def _error_unicode_in_app_iter(app_iter, body):
    app_iter_repr = repr(app_iter)
    if len(app_iter_repr) > 50:
        app_iter_repr = app_iter_repr[:30] + '...' + app_iter_repr[-10:]
    raise TypeError('An item of the app_iter (%s) was text, causing a text body: %r' % (app_iter_repr, body))