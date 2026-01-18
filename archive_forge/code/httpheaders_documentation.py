import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest

        Return a list of language tags sorted by their "q" values.  For example,
        "en-us,en;q=0.5" should return ``["en-us", "en"]``.  If there is no
        ``Accept-Language`` header present, default to ``[]``.
        