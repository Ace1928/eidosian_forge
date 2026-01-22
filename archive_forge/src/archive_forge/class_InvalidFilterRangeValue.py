import urllib.parse as urlparse
from glance.i18n import _
class InvalidFilterRangeValue(Invalid):
    message = _('Unable to filter using the specified range.')