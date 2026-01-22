import logging
from urllib.parse import urljoin, urlparse
from w3lib.url import safe_url_string
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import HtmlResponse
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.response import get_meta_refresh
def _build_redirect_request(source_request, *, url, **kwargs):
    redirect_request = source_request.replace(url=url, **kwargs, cookies=None)
    has_cookie_header = 'Cookie' in redirect_request.headers
    has_authorization_header = 'Authorization' in redirect_request.headers
    if has_cookie_header or has_authorization_header:
        source_request_netloc = urlparse_cached(source_request).netloc
        redirect_request_netloc = urlparse_cached(redirect_request).netloc
        if source_request_netloc != redirect_request_netloc:
            if has_cookie_header:
                del redirect_request.headers['Cookie']
            if has_authorization_header:
                del redirect_request.headers['Authorization']
    return redirect_request