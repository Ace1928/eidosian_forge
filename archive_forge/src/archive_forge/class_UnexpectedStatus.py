import urllib.parse as urlparse
from glance.i18n import _
class UnexpectedStatus(GlanceException):
    message = _('The request returned an unexpected status: %(status)s.\n\nThe response body:\n%(body)s')