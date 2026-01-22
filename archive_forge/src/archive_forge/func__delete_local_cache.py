from dogpile.cache import api
from dogpile.cache import proxy
from oslo_context import context as oslo_context
from oslo_serialization import msgpackutils
def _delete_local_cache(self, key):
    ctx = self._get_request_context()
    try:
        delattr(ctx, self._get_request_key(key))
    except AttributeError:
        pass