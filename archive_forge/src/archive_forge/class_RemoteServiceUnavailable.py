import urllib.parse
from glance_store.i18n import _
class RemoteServiceUnavailable(GlanceStoreException):
    message = _('Remote server where the image is present is unavailable.')