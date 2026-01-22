import urllib.parse
from glance_store.i18n import _
class StoreRandomGetNotSupported(StoreGetNotSupported):
    message = _('Getting images randomly from this store is not supported. Offset: %(offset)s, length: %(chunk_size)s')