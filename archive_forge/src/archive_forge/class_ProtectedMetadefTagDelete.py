import urllib.parse as urlparse
from glance.i18n import _
class ProtectedMetadefTagDelete(Forbidden):
    message = _('Metadata definition tag %(tag_name)s is protected and cannot be deleted.')