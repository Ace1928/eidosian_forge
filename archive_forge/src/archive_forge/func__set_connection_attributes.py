import re
from time import time
from typing import Optional, Tuple
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from twisted.internet import defer
from twisted.internet.protocol import ClientFactory
from twisted.web.http import HTTPClient
from scrapy import Request
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes, to_unicode
def _set_connection_attributes(self, request):
    parsed = urlparse_cached(request)
    self.scheme, self.netloc, self.host, self.port, self.path = _parsed_url_args(parsed)
    proxy = request.meta.get('proxy')
    if proxy:
        self.scheme, _, self.host, self.port, _ = _parse(proxy)
        self.path = self.url