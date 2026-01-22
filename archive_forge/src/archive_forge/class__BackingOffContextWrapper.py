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
class _BackingOffContextWrapper(_ContextWrapper):
    """Wraps oslo messaging contexts to set the timeout for calls.

    This intercepts RPC calls and sets the timeout value to the globally
    adapting value for each method. An oslo messaging timeout results in
    a doubling of the timeout value for the method on which it timed out.
    There currently is no logic to reduce the timeout since busy Neutron
    servers are more frequently the cause of timeouts rather than lost
    messages.
    """
    _METHOD_TIMEOUTS = _get_default_method_timeouts()
    _max_timeout = None

    @classmethod
    def reset_timeouts(cls):
        cls._METHOD_TIMEOUTS = _get_default_method_timeouts()
        cls._max_timeout = None

    @classmethod
    def get_max_timeout(cls):
        return cls._max_timeout or _get_rpc_response_max_timeout()

    @classmethod
    def set_max_timeout(cls, max_timeout):
        if max_timeout < cls.get_max_timeout():
            cls._METHOD_TIMEOUTS.default_factory = lambda: max_timeout
            for k, v in cls._METHOD_TIMEOUTS.items():
                if v > max_timeout:
                    cls._METHOD_TIMEOUTS[k] = max_timeout
            cls._max_timeout = max_timeout

    def call(self, ctxt, method, **kwargs):
        if self._original_context.target.namespace:
            scoped_method = '%s.%s' % (self._original_context.target.namespace, method)
        else:
            scoped_method = method
        self._original_context.timeout = self._METHOD_TIMEOUTS[scoped_method]
        try:
            return self._original_context.call(ctxt, method, **kwargs)
        except oslo_messaging.MessagingTimeout:
            with excutils.save_and_reraise_exception():
                wait = random.uniform(0, min(self._METHOD_TIMEOUTS[scoped_method], TRANSPORT.conf.rpc_response_timeout))
                LOG.error('Timeout in RPC method %(method)s. Waiting for %(wait)s seconds before next attempt. If the server is not down, consider increasing the rpc_response_timeout option as Neutron server(s) may be overloaded and unable to respond quickly enough.', {'wait': int(round(wait)), 'method': scoped_method})
                new_timeout = min(self._original_context.timeout * 2, self.get_max_timeout())
                if new_timeout > self._METHOD_TIMEOUTS[scoped_method]:
                    LOG.warning('Increasing timeout for %(method)s calls to %(new)s seconds. Restart the agent to restore it to the default value.', {'method': scoped_method, 'new': new_timeout})
                    self._METHOD_TIMEOUTS[scoped_method] = new_timeout
                time.sleep(wait)