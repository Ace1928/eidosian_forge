import urllib.parse
from glance_store.i18n import _
class StoreAddDisabled(GlanceStoreException):
    message = _('Configuration for store failed. Adding images to this store is disabled.')