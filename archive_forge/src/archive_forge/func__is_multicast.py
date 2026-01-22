import abc
import logging
from oslo_messaging.target import Target
def _is_multicast(self, address):
    return address.startswith(self._rpc_multicast) or address.startswith(self._notify_multicast)