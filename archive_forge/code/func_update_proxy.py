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
def update_proxy(self, proxy: Optional[URL], proxy_auth: Optional[BasicAuth], proxy_headers: Optional[LooseHeaders]) -> None:
    if proxy_auth and (not isinstance(proxy_auth, helpers.BasicAuth)):
        raise ValueError('proxy_auth must be None or BasicAuth() tuple')
    self.proxy = proxy
    self.proxy_auth = proxy_auth
    self.proxy_headers = proxy_headers