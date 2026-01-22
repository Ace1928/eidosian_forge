from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork, NetworkRoute
def _get_route(self):
    destination = self.module.params.get('destination')
    gateway = self.module.params.get('gateway')
    for route in self.hcloud_network.routes:
        if route.destination == destination and route.gateway == gateway:
            self.hcloud_route = route