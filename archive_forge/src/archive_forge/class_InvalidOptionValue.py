import urllib.parse as urlparse
from glance.i18n import _
class InvalidOptionValue(Invalid):
    message = _('Invalid value for option %(option)s: %(value)s')