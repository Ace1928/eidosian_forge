import re
import time
from http.cookiejar import CookieJar as _CookieJar
from http.cookiejar import DefaultCookiePolicy
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
class WrappedRequest:
    """Wraps a scrapy Request class with methods defined by urllib2.Request class to interact with CookieJar class

    see http://docs.python.org/library/urllib2.html#urllib2.Request
    """

    def __init__(self, request):
        self.request = request

    def get_full_url(self):
        return self.request.url

    def get_host(self):
        return urlparse_cached(self.request).netloc

    def get_type(self):
        return urlparse_cached(self.request).scheme

    def is_unverifiable(self):
        """Unverifiable should indicate whether the request is unverifiable, as defined by RFC 2965.

        It defaults to False. An unverifiable request is one whose URL the user did not have the
        option to approve. For example, if the request is for an image in an
        HTML document, and the user had no option to approve the automatic
        fetching of the image, this should be true.
        """
        return self.request.meta.get('is_unverifiable', False)

    @property
    def full_url(self):
        return self.get_full_url()

    @property
    def host(self):
        return self.get_host()

    @property
    def type(self):
        return self.get_type()

    @property
    def unverifiable(self):
        return self.is_unverifiable()

    @property
    def origin_req_host(self):
        return urlparse_cached(self.request).hostname

    def has_header(self, name):
        return name in self.request.headers

    def get_header(self, name, default=None):
        return to_unicode(self.request.headers.get(name, default), errors='replace')

    def header_items(self):
        return [(to_unicode(k, errors='replace'), [to_unicode(x, errors='replace') for x in v]) for k, v in self.request.headers.items()]

    def add_unredirected_header(self, name, value):
        self.request.headers.appendlist(name, value)