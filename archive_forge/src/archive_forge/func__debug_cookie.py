import logging
from collections import defaultdict
from tldextract import TLDExtract
from scrapy.exceptions import NotConfigured
from scrapy.http import Response
from scrapy.http.cookies import CookieJar
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
def _debug_cookie(self, request, spider):
    if self.debug:
        cl = [to_unicode(c, errors='replace') for c in request.headers.getlist('Cookie')]
        if cl:
            cookies = '\n'.join((f'Cookie: {c}\n' for c in cl))
            msg = f'Sending cookies to: {request}\n{cookies}'
            logger.debug(msg, extra={'spider': spider})