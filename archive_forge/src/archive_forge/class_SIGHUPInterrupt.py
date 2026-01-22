import urllib.parse as urlparse
from glance.i18n import _
class SIGHUPInterrupt(GlanceException):
    message = _('System SIGHUP signal received.')