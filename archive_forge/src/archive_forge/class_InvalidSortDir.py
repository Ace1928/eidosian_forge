import urllib.parse as urlparse
from glance.i18n import _
class InvalidSortDir(Invalid):
    message = _('Sort direction supplied was not valid.')