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
def _to_instancegroup(self, instancegroup):
    """
        Return the InstanceGroup object from the JSON-response.

        :param  instancegroup:  Dictionary describing InstanceGroup
        :type   instancegroup: ``dict``

        :return: InstanceGroup object.
        :rtype: :class:`GCEInstanceGroup`
        """
    extra = {}
    extra['description'] = instancegroup.get('description', None)
    extra['selfLink'] = instancegroup['selfLink']
    extra['namedPorts'] = instancegroup.get('namedPorts', [])
    extra['fingerprint'] = instancegroup.get('fingerprint', None)
    zone = instancegroup.get('zone', None)
    if zone:
        zone = self.ex_get_zone(zone)
    network = instancegroup.get('network', None)
    if network:
        network = self.ex_get_network(network)
    subnetwork = instancegroup.get('subnetwork', None)
    if subnetwork:
        subnetwork = self.ex_get_subnetwork(subnetwork)
    return GCEInstanceGroup(id=instancegroup['id'], name=instancegroup['name'], zone=zone, network=network, subnetwork=subnetwork, named_ports=instancegroup.get('namedPorts', []), driver=self, extra=extra)