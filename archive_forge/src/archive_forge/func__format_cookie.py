import logging
from collections import defaultdict
from tldextract import TLDExtract
from scrapy.exceptions import NotConfigured
from scrapy.http import Response
from scrapy.http.cookies import CookieJar
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
def _format_cookie(self, cookie, request):
    """
        Given a dict consisting of cookie components, return its string representation.
        Decode from bytes if necessary.
        """
    decoded = {}
    for key in ('name', 'value', 'path', 'domain'):
        if cookie.get(key) is None:
            if key in ('name', 'value'):
                msg = f"Invalid cookie found in request {request}: {cookie} ('{key}' is missing)"
                logger.warning(msg)
                return
            continue
        if isinstance(cookie[key], (bool, float, int, str)):
            decoded[key] = str(cookie[key])
        else:
            try:
                decoded[key] = cookie[key].decode('utf8')
            except UnicodeDecodeError:
                logger.warning('Non UTF-8 encoded cookie found in request %s: %s', request, cookie)
                decoded[key] = cookie[key].decode('latin1', errors='replace')
    cookie_str = f'{decoded.pop('name')}={decoded.pop('value')}'
    for key, value in decoded.items():
        cookie_str += f'; {key.capitalize()}={value}'
    return cookie_str