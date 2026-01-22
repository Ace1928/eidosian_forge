import urllib.parse as urlparse
from glance.i18n import _
class SchemaLoadError(GlanceException):
    message = _('Unable to load schema: %(reason)s')