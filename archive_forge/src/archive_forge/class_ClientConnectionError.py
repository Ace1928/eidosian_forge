import urllib.parse as urlparse
from glance.i18n import _
class ClientConnectionError(GlanceException):
    message = _('There was an error connecting to a server')