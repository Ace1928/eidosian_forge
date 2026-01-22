import urllib.parse
from glance_store.i18n import _
class InUseByStore(GlanceStoreException):
    message = _('The image cannot be deleted because it is in use through the backend store outside of Glance.')