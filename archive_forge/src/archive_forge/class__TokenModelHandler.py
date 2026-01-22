from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
class _TokenModelHandler(object):
    identity = 126
    handles = (TokenModel,)

    def __init__(self, registry):
        self._registry = registry

    def serialize(self, obj):
        serialized = msgpackutils.dumps(obj.__dict__, registry=self._registry)
        return serialized

    def deserialize(self, data):
        token_data = msgpackutils.loads(data, registry=self._registry)
        try:
            token_model = TokenModel()
            for k, v in iter(token_data.items()):
                setattr(token_model, k, v)
        except Exception:
            LOG.debug('Failed to deserialize TokenModel. Data is %s', token_data)
            raise exception.CacheDeserializationError(TokenModel.__name__, token_data)
        return token_model