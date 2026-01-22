from dogpile.cache import api
from dogpile.cache import proxy
from oslo_context import context as oslo_context
from oslo_serialization import msgpackutils
def _set_local_cache(self, key, value):
    ctx = self._get_request_context()
    serialize = {'payload': value.payload, 'metadata': value.metadata}
    setattr(ctx, self._get_request_key(key), msgpackutils.dumps(serialize))