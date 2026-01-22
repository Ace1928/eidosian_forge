import urllib.parse as urlparse
from glance.i18n import _
class InvalidPropertyProtectionConfiguration(Invalid):
    message = _('Invalid configuration in property protection file.')