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
def _request_uri(environ):
    """Like ``wsgiref.url.request_uri``, except eliminates ``:80`` ports.

    Returns the full request URI."""
    url = environ['wsgi.url_scheme'] + '://'
    if environ.get('HTTP_HOST'):
        url += environ['HTTP_HOST']
    else:
        url += environ['SERVER_NAME'] + ':' + environ['SERVER_PORT']
    if url.endswith(':80') and environ['wsgi.url_scheme'] == 'http':
        url = url[:-3]
    elif url.endswith(':443') and environ['wsgi.url_scheme'] == 'https':
        url = url[:-4]
    if PY2:
        script_name = environ.get('SCRIPT_NAME', '/')
        path_info = environ.get('PATH_INFO', '')
    else:
        script_name = bytes_(environ.get('SCRIPT_NAME', '/'), 'latin-1')
        path_info = bytes_(environ.get('PATH_INFO', ''), 'latin-1')
    url += url_quote(script_name)
    qpath_info = url_quote(path_info)
    if 'SCRIPT_NAME' not in environ:
        url += qpath_info[1:]
    else:
        url += qpath_info
    return url