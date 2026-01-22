import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def _find_floating_network_by_router(self):
    """Find the network providing floating ips by looking at routers."""
    for router in self.list_routers():
        if router['admin_state_up']:
            network_id = router.get('external_gateway_info', {}).get('network_id')
            if network_id:
                return network_id