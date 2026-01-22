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
def _has_body__get(self):
    """
        Determine if the the response has a :attr:`~Response.body`. In
        contrast to simply accessing :attr:`~Response.body`, this method
        will **not** read the underlying :attr:`~Response.app_iter`.
        """
    app_iter = self._app_iter
    if isinstance(app_iter, list) and len(app_iter) == 1:
        if app_iter[0] != b'':
            return True
        else:
            return False
    if app_iter is None:
        return False
    return True