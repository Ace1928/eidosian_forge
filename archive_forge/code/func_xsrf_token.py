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
@property
def xsrf_token(self) -> bytes:
    """The XSRF-prevention token for the current user/session.

        To prevent cross-site request forgery, we set an '_xsrf' cookie
        and include the same '_xsrf' value as an argument with all POST
        requests. If the two do not match, we reject the form submission
        as a potential forgery.

        See http://en.wikipedia.org/wiki/Cross-site_request_forgery

        This property is of type `bytes`, but it contains only ASCII
        characters. If a character string is required, there is no
        need to base64-encode it; just decode the byte string as
        UTF-8.

        .. versionchanged:: 3.2.2
           The xsrf token will now be have a random mask applied in every
           request, which makes it safe to include the token in pages
           that are compressed.  See http://breachattack.com for more
           information on the issue fixed by this change.  Old (version 1)
           cookies will be converted to version 2 when this method is called
           unless the ``xsrf_cookie_version`` `Application` setting is
           set to 1.

        .. versionchanged:: 4.3
           The ``xsrf_cookie_kwargs`` `Application` setting may be
           used to supply additional cookie options (which will be
           passed directly to `set_cookie`). For example,
           ``xsrf_cookie_kwargs=dict(httponly=True, secure=True)``
           will set the ``secure`` and ``httponly`` flags on the
           ``_xsrf`` cookie.
        """
    if not hasattr(self, '_xsrf_token'):
        version, token, timestamp = self._get_raw_xsrf_token()
        output_version = self.settings.get('xsrf_cookie_version', 2)
        cookie_kwargs = self.settings.get('xsrf_cookie_kwargs', {})
        if output_version == 1:
            self._xsrf_token = binascii.b2a_hex(token)
        elif output_version == 2:
            mask = os.urandom(4)
            self._xsrf_token = b'|'.join([b'2', binascii.b2a_hex(mask), binascii.b2a_hex(_websocket_mask(mask, token)), utf8(str(int(timestamp)))])
        else:
            raise ValueError('unknown xsrf cookie version %d', output_version)
        if version is None:
            if self.current_user and 'expires_days' not in cookie_kwargs:
                cookie_kwargs['expires_days'] = 30
            cookie_name = self.settings.get('xsrf_cookie_name', '_xsrf')
            self.set_cookie(cookie_name, self._xsrf_token, **cookie_kwargs)
    return self._xsrf_token