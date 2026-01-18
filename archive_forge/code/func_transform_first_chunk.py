import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import (
from tornado.util import ObjectDict, unicode_type, _websocket_mask
from typing import (
from types import TracebackType
import typing
def transform_first_chunk(self, status_code: int, headers: httputil.HTTPHeaders, chunk: bytes, finishing: bool) -> Tuple[int, httputil.HTTPHeaders, bytes]:
    if 'Vary' in headers:
        headers['Vary'] += ', Accept-Encoding'
    else:
        headers['Vary'] = 'Accept-Encoding'
    if self._gzipping:
        ctype = _unicode(headers.get('Content-Type', '')).split(';')[0]
        self._gzipping = self._compressible_type(ctype) and (not finishing or len(chunk) >= self.MIN_LENGTH) and ('Content-Encoding' not in headers)
    if self._gzipping:
        headers['Content-Encoding'] = 'gzip'
        self._gzip_value = BytesIO()
        self._gzip_file = gzip.GzipFile(mode='w', fileobj=self._gzip_value, compresslevel=self.GZIP_LEVEL)
        chunk = self.transform_chunk(chunk, finishing)
        if 'Content-Length' in headers:
            if finishing:
                headers['Content-Length'] = str(len(chunk))
            else:
                del headers['Content-Length']
    return (status_code, headers, chunk)