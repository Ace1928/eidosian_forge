import urllib.parse as urlparse
from glance.i18n import _
class NoServiceEndpoint(GlanceException):
    message = _('Response from Keystone does not contain a Glance endpoint.')