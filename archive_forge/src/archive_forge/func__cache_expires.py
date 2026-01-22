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
def _cache_expires(self, seconds=0, **kw):
    """
            Set expiration on this request.  This sets the response to
            expire in the given seconds, and any other attributes are used
            for ``cache_control`` (e.g., ``private=True``).
        """
    if seconds is True:
        seconds = 0
    elif isinstance(seconds, timedelta):
        seconds = timedelta_to_seconds(seconds)
    cache_control = self.cache_control
    if seconds is None:
        pass
    elif not seconds:
        cache_control.no_store = True
        cache_control.no_cache = True
        cache_control.must_revalidate = True
        cache_control.max_age = 0
        cache_control.post_check = 0
        cache_control.pre_check = 0
        self.expires = datetime.utcnow()
        if 'last-modified' not in self.headers:
            self.last_modified = datetime.utcnow()
        self.pragma = 'no-cache'
    else:
        cache_control.properties.clear()
        cache_control.max_age = seconds
        self.expires = datetime.utcnow() + timedelta(seconds=seconds)
        self.pragma = None
    for name, value in kw.items():
        setattr(cache_control, name, value)