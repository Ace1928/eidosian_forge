from oslo_config import cfg
import oslo_messaging as messaging
from heat.common import messaging as rpc_messaging
from heat.rpc import api as rpc_api
class EngineListenerClient(object):
    """Client side of the heat listener RPC API.

    API version history::

        1.0 - Initial version.
    """
    BASE_RPC_API_VERSION = '1.0'

    def __init__(self, engine_id):
        _client = rpc_messaging.get_rpc_client(topic=rpc_api.LISTENER_TOPIC, version=self.BASE_RPC_API_VERSION, server=engine_id)
        self._client = _client.prepare(timeout=cfg.CONF.engine_life_check_timeout)

    def is_alive(self, ctxt):
        try:
            return self._client.call(ctxt, 'listening')
        except messaging.MessagingTimeout:
            return False