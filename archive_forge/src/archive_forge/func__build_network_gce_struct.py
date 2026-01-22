import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _build_network_gce_struct(self, network, subnetwork=None, external_ip=None, use_selflinks=True, internal_ip=None):
    """
        Build network interface dict for use in the GCE API.

        Note: Must be wrapped in a list before passing to the GCE API.

        :param    network: The network to associate with the node.
        :type     network: :class:`GCENetwork`

        :keyword  subnetwork: The subnetwork to include.
        :type     subnetwork: :class:`GCESubNetwork`

        :keyword  external_ip: The external IP address to use.  If 'ephemeral'
                               (default), a new non-static address will be
                               used.  If 'None', then no external address will
                               be used.  To use an existing static IP address,
                               a GCEAddress object should be passed in.
        :type     external_ip: :class:`GCEAddress`

        :keyword  internal_ip: The private IP address to use.
        :type     internal_ip: :class:`GCEAddress` or ``str``

        :return:  network interface dict
        :rtype:   ``dict``
        """
    ni = {}
    ni = {'kind': 'compute#instanceNetworkInterface'}
    if network is None:
        network = 'default'
    ni['network'] = self._get_selflink_or_name(obj=network, get_selflinks=use_selflinks, objname='network')
    if subnetwork:
        ni['subnetwork'] = self._get_selflink_or_name(obj=subnetwork, get_selflinks=use_selflinks, objname='subnetwork')
    if external_ip:
        access_configs = [{'name': 'External NAT', 'type': 'ONE_TO_ONE_NAT'}]
        if hasattr(external_ip, 'address'):
            access_configs[0]['natIP'] = external_ip.address
        ni['accessConfigs'] = access_configs
    if internal_ip:
        ni['networkIP'] = internal_ip
    return ni