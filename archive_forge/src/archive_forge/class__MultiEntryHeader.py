import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _MultiEntryHeader(HTTPHeader):
    """
    a multi-value ``HTTPHeader`` where items cannot be combined with a comma

    This header is multi-valued, but the values should not be combined
    with a comma since the header is not in compliance with RFC 2616
    (Set-Cookie due to Expires parameter) or which common user-agents do
    not behave well when the header values are combined.
    """

    def update(self, collection, *args, **kwargs):
        assert list == type(collection), '``environ`` may not be updated'
        self.delete(collection)
        collection.extend(self.tuples(*args, **kwargs))

    def tuples(self, *args, **kwargs):
        values = self.values(*args, **kwargs)
        if not values:
            return ()
        return [(self.name, value.strip()) for value in values]