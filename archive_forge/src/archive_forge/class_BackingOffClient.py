import collections
import random
import time
from neutron_lib._i18n import _
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.utils import runtime
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_messaging import exceptions as oslomsg_exc
from oslo_messaging import serializer as om_serializer
from oslo_service import service
from oslo_utils import excutils
from osprofiler import profiler
class BackingOffClient(oslo_messaging.RPCClient):
    """An oslo messaging RPC Client that implements a timeout backoff.

    This has all of the same interfaces as oslo_messaging.RPCClient but
    if the timeout parameter is not specified, the _BackingOffContextWrapper
    returned will track when call timeout exceptions occur and exponentially
    increase the timeout for the given call method.
    """

    def prepare(self, *args, **kwargs):
        ctx = super().prepare(*args, **kwargs)
        if 'timeout' in kwargs:
            return _ContextWrapper(ctx)
        return _BackingOffContextWrapper(ctx)

    @staticmethod
    def set_max_timeout(max_timeout):
        """Set RPC timeout ceiling for all backing-off RPC clients."""
        _BackingOffContextWrapper.set_max_timeout(max_timeout)