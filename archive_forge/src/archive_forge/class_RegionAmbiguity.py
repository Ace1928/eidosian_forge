import urllib.parse as urlparse
from glance.i18n import _
class RegionAmbiguity(GlanceException):
    message = _("Multiple 'image' service matches for region %(region)s. This generally means that a region is required and you have not supplied one.")