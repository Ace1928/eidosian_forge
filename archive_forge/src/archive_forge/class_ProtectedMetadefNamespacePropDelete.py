import urllib.parse as urlparse
from glance.i18n import _
class ProtectedMetadefNamespacePropDelete(Forbidden):
    message = _('Metadata definition property %(property_name)s is protected and cannot be deleted.')