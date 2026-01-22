import logging
from urllib.parse import urljoin, urlparse
from w3lib.url import safe_url_string
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import HtmlResponse
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.response import get_meta_refresh
def _redirect_request_using_get(self, request, redirect_url):
    redirect_request = _build_redirect_request(request, url=redirect_url, method='GET', body='')
    redirect_request.headers.pop('Content-Type', None)
    redirect_request.headers.pop('Content-Length', None)
    return redirect_request