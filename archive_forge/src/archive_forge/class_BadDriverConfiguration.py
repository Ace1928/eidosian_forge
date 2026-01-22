import urllib.parse as urlparse
from glance.i18n import _
class BadDriverConfiguration(GlanceException):
    message = _('Driver %(driver_name)s could not be configured correctly. Reason: %(reason)s')