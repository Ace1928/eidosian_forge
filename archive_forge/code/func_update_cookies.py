import asyncio
import codecs
import contextlib
import functools
import io
import re
import sys
import traceback
import warnings
from hashlib import md5, sha1, sha256
from http.cookies import CookieError, Morsel, SimpleCookie
from types import MappingProxyType, TracebackType
from typing import (
import attr
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .helpers import (
from .http import (
from .log import client_logger
from .streams import StreamReader
from .typedefs import (
def update_cookies(self, cookies: Optional[LooseCookies]) -> None:
    """Update request cookies header."""
    if not cookies:
        return
    c = SimpleCookie()
    if hdrs.COOKIE in self.headers:
        c.load(self.headers.get(hdrs.COOKIE, ''))
        del self.headers[hdrs.COOKIE]
    if isinstance(cookies, Mapping):
        iter_cookies = cookies.items()
    else:
        iter_cookies = cookies
    for name, value in iter_cookies:
        if isinstance(value, Morsel):
            mrsl_val = value.get(value.key, Morsel())
            mrsl_val.set(value.key, value.value, value.coded_value)
            c[name] = mrsl_val
        else:
            c[name] = value
    self.headers[hdrs.COOKIE] = c.output(header='', sep=';').strip()