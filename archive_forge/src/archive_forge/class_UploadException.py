import urllib.parse as urlparse
from glance.i18n import _
class UploadException(GlanceException):
    message = _('Image upload problem: %s')