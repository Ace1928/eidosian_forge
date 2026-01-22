from webob.datetime_utils import (
from webob.descriptors import _rx_etag
from webob.util import header_docstring
class _NoETag(object):
    """
    Represents a missing ETag when matching is unsafe
    """

    def __repr__(self):
        return '<No ETag>'

    def __nonzero__(self):
        return False
    __bool__ = __nonzero__

    def __contains__(self, other):
        return False

    def __str__(self):
        return ''