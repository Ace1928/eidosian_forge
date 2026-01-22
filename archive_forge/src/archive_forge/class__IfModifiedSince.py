import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _IfModifiedSince(_DateHeader):
    """
    If-Modified-Since, RFC 2616 section 14.25
    """
    version = '1.0'

    def __call__(self, *args, **kwargs):
        """
        Split the value on ';' incase the header includes extra attributes. E.g.
        IE 6 is known to send:
        If-Modified-Since: Sun, 25 Jun 2006 20:36:35 GMT; length=1506
        """
        return _DateHeader.__call__(self, *args, **kwargs).split(';', 1)[0]

    def parse(self, *args, **kwargs):
        value = _DateHeader.parse(self, *args, **kwargs)
        if value and value > now():
            raise HTTPBadRequest('Please check your system clock.\r\nAccording to this server, the time provided in the\r\n%s header is in the future.\r\n' % self.name)
        return value