import logging
from collections import defaultdict
from tldextract import TLDExtract
from scrapy.exceptions import NotConfigured
from scrapy.http import Response
from scrapy.http.cookies import CookieJar
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
def _is_public_domain(domain):
    parts = _split_domain(domain)
    return not parts.domain