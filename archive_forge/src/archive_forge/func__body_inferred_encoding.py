from __future__ import annotations
import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Generator, Optional, Tuple
from urllib.parse import urljoin
import parsel
from w3lib.encoding import (
from w3lib.html import strip_html5_whitespace
from scrapy.http import Request
from scrapy.http.response import Response
from scrapy.utils.python import memoizemethod_noargs, to_unicode
from scrapy.utils.response import get_base_url
def _body_inferred_encoding(self):
    if self._cached_benc is None:
        content_type = to_unicode(self.headers.get(b'Content-Type', b''), encoding='latin-1')
        benc, ubody = html_to_unicode(content_type, self.body, auto_detect_fun=self._auto_detect_fun, default_encoding=self._DEFAULT_ENCODING)
        self._cached_benc = benc
        self._cached_ubody = ubody
    return self._cached_benc