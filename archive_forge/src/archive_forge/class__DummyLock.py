import re
import time
from http.cookiejar import CookieJar as _CookieJar
from http.cookiejar import DefaultCookiePolicy
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
class _DummyLock:

    def acquire(self):
        pass

    def release(self):
        pass