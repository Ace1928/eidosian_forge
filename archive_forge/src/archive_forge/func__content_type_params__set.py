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
def _content_type_params__set(self, value_dict):
    if not value_dict:
        self._content_type_params__del()
        return
    params = []
    for k, v in sorted(value_dict.items()):
        if not _OK_PARAM_RE.search(v):
            v = '"%s"' % v.replace('"', '\\"')
        params.append('; %s=%s' % (k, v))
    ct = self.headers.pop('Content-Type', '').split(';', 1)[0]
    ct += ''.join(params)
    self.headers['Content-Type'] = ct