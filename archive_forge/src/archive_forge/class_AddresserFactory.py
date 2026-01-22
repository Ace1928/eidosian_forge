import abc
import logging
from oslo_messaging.target import Target
class AddresserFactory(object):
    """Generates the proper Addresser based on configuration and the type of
    message bus the driver is connected to.
    """

    def __init__(self, default_exchange, mode, **kwargs):
        self._default_exchange = default_exchange
        self._mode = mode
        self._kwargs = kwargs

    def __call__(self, remote_properties, vhost=None):
        product = remote_properties.get('product', 'qpid-cpp')
        if self._mode == 'legacy' or (self._mode == 'dynamic' and product == 'qpid-cpp'):
            return LegacyAddresser(self._default_exchange, self._kwargs['legacy_server_prefix'], self._kwargs['legacy_broadcast_prefix'], self._kwargs['legacy_group_prefix'], vhost)
        else:
            return RoutableAddresser(self._default_exchange, self._kwargs.get('rpc_exchange'), self._kwargs['rpc_prefix'], self._kwargs.get('notify_exchange'), self._kwargs['notify_prefix'], self._kwargs['unicast'], self._kwargs['multicast'], self._kwargs['anycast'], vhost)