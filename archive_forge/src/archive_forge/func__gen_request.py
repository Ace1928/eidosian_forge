import sys
import io
import random
import mimetypes
import time
import os
import shutil
import smtplib
import shlex
import re
import subprocess
from urllib.parse import urlencode
from urllib import parse as urlparse
from http.cookies import BaseCookie
from paste import wsgilib
from paste import lint
from paste.response import HeaderDict
def _gen_request(self, method, url, params=b'', headers=None, extra_environ=None, status=None, upload_files=None, expect_errors=False):
    """
        Do a generic request.
        """
    if headers is None:
        headers = {}
    if extra_environ is None:
        extra_environ = {}
    environ = self._make_environ()
    if isinstance(params, (list, tuple, dict)):
        params = urlencode(params)
    if hasattr(params, 'items'):
        params = urlencode(params.items())
    if isinstance(params, str):
        params = params.encode('utf8')
    if upload_files:
        params = urlparse.parse_qsl(params, keep_blank_values=True)
        content_type, params = self.encode_multipart(params, upload_files)
        environ['CONTENT_TYPE'] = content_type
    elif params:
        environ.setdefault('CONTENT_TYPE', 'application/x-www-form-urlencoded')
    url = str(url)
    if '?' in url:
        url, environ['QUERY_STRING'] = url.split('?', 1)
    else:
        environ['QUERY_STRING'] = ''
    environ['CONTENT_LENGTH'] = str(len(params))
    environ['REQUEST_METHOD'] = method
    environ['wsgi.input'] = io.BytesIO(params)
    self._set_headers(headers, environ)
    environ.update(extra_environ)
    req = TestRequest(url, environ, expect_errors)
    return self.do_request(req, status=status)