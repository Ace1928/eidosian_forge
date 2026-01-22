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
def _multi_create_node(self, status, node_attrs):
    """Create node for ex_create_multiple_nodes.

        :param  status: Dictionary for holding node creation status.
                        (This dictionary is modified by this method)
        :type   status: ``dict``

        :param  node_attrs: Dictionary for holding node attribute information.
                            (size, image, location, etc.)
        :type   node_attrs: ``dict``
        """
    request, node_data = self._create_node_req(status['name'], node_attrs['size'], node_attrs['image'], node_attrs['location'], node_attrs['network'], node_attrs['tags'], node_attrs['metadata'], external_ip=node_attrs['external_ip'], internal_ip=node_attrs['internal_ip'], ex_service_accounts=node_attrs['ex_service_accounts'], description=node_attrs['description'], ex_can_ip_forward=node_attrs['ex_can_ip_forward'], ex_disk_auto_delete=node_attrs['ex_disk_auto_delete'], ex_disks_gce_struct=node_attrs['ex_disks_gce_struct'], ex_nic_gce_struct=node_attrs['ex_nic_gce_struct'], ex_on_host_maintenance=node_attrs['ex_on_host_maintenance'], ex_automatic_restart=node_attrs['ex_automatic_restart'], ex_subnetwork=node_attrs['subnetwork'], ex_preemptible=node_attrs['ex_preemptible'], ex_labels=node_attrs['ex_labels'], ex_disk_size=node_attrs['ex_disk_size'])
    try:
        node_res = self.connection.request(request, method='POST', data=node_data).object
    except GoogleBaseError:
        e = self._catch_error(ignore_errors=node_attrs['ignore_errors'])
        error = e.value
        code = e.code
        node_res = None
        status['node'] = GCEFailedNode(status['name'], error, code)
    status['node_response'] = node_res