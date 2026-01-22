import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _AcceptLanguage(_MultiValueHeader):
    """
    Accept-Language, RFC 2616 section 14.4
    """

    def parse(self, *args, **kwargs):
        """
        Return a list of language tags sorted by their "q" values.  For example,
        "en-us,en;q=0.5" should return ``["en-us", "en"]``.  If there is no
        ``Accept-Language`` header present, default to ``[]``.
        """
        header = self.__call__(*args, **kwargs)
        if header is None:
            return []
        langs = [v for v in header.split(',') if v]
        qs = []
        for lang in langs:
            pieces = lang.split(';')
            lang, params = (pieces[0].strip().lower(), pieces[1:])
            q = 1
            for param in params:
                if '=' not in param:
                    continue
                lvalue, rvalue = param.split('=')
                lvalue = lvalue.strip().lower()
                rvalue = rvalue.strip()
                if lvalue == 'q':
                    q = float(rvalue)
            qs.append((lang, q))
        qs.sort(key=lambda query: query[1], reverse=True)
        return [lang for lang, q in qs]