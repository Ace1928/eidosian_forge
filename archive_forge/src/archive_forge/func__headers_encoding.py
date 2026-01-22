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
@memoizemethod_noargs
def _headers_encoding(self):
    content_type = self.headers.get(b'Content-Type', b'')
    return http_content_type_encoding(to_unicode(content_type, encoding='latin-1'))