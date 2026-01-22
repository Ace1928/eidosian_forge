import urllib.parse as urlparse
from glance.i18n import _
class ClientConfigurationError(GlanceException):
    message = _('There was an error configuring the client.')