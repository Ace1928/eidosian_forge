import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
@property
def _network_type(self):
    """Discover whether the running cloud is using neutron or nova-network.

        :return:
            * ``NET_TYPE_NEUTRON`` if neutron is detected
            * ``NET_TYPE_COMPUTE`` if running in a cloud but neutron is not
              detected.
            * ``None`` if not running in a cloud, which hopefully means we're
              building docs.
        """
    if not hasattr(self, '_net_type'):
        try:
            if self.app.client_manager.is_network_endpoint_enabled():
                net_type = _NET_TYPE_NEUTRON
            else:
                net_type = _NET_TYPE_COMPUTE
        except AttributeError:
            LOG.warning('%s: Could not detect a network type. Assuming we are building docs.', self.__class__.__name__)
            net_type = None
        self._net_type = net_type
    return self._net_type