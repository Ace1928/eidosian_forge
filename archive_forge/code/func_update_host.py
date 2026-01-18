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
def update_host(self, url: URL) -> None:
    """Update destination host, port and connection type (ssl)."""
    if not url.raw_host:
        raise InvalidURL(url)
    username, password = (url.user, url.password)
    if username:
        self.auth = helpers.BasicAuth(username, password or '')