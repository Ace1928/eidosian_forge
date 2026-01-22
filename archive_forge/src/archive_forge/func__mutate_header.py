import base64
import binascii
import hashlib
import hmac
import json
from datetime import (
import re
import string
import time
import warnings
from webob.compat import (
from webob.util import strings_differ
def _mutate_header(self, name, value):
    header = self._environ.get('HTTP_COOKIE')
    had_header = header is not None
    header = header or ''
    if not PY2:
        header = header.encode('latin-1')
    bytes_name = bytes_(name, 'ascii')
    if value is None:
        replacement = None
    else:
        bytes_val = _value_quote(bytes_(value, 'utf-8'))
        replacement = bytes_name + b'=' + bytes_val
    matches = _rx_cookie.finditer(header)
    found = False
    for match in matches:
        start, end = match.span()
        match_name = match.group(1)
        if match_name == bytes_name:
            found = True
            if replacement is None:
                header = header[:start].rstrip(b' ;') + header[end:]
            else:
                header = header[:start] + replacement + header[end:]
            break
    else:
        if replacement is not None:
            if header:
                header += b'; ' + replacement
            else:
                header = replacement
    if header:
        self._environ['HTTP_COOKIE'] = native_(header, 'latin-1')
    elif had_header:
        self._environ['HTTP_COOKIE'] = ''
    return found