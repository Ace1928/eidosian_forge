from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork, NetworkSubnet
def _get_subnetwork(self):
    subnet_ip_range = self.module.params.get('ip_range')
    for subnetwork in self.hcloud_network.subnets:
        if subnetwork.ip_range == subnet_ip_range:
            self.hcloud_subnetwork = subnetwork