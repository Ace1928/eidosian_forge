import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _AcceptRanges(_MultiValueHeader):
    """
    Accept-Ranges, RFC 2616 section 14.5
    """

    def compose(self, none=None, bytes=None):
        if bytes:
            return ('bytes',)
        return ('none',)