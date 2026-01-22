import urllib.parse as urlparse
from glance.i18n import _
class StorageQuotaFull(GlanceException):
    message = _('The size of the data %(image_size)s will exceed the limit. %(remaining)s bytes remaining.')