from oslo_config import cfg
import oslo_messaging as messaging
from heat.common import messaging as rpc_messaging
from heat.rpc import api as rpc_api
Client side of the heat listener RPC API.

    API version history::

        1.0 - Initial version.
    