from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer, PrivateNet
from ..module_utils.vendor.hcloud.networks import BoundNetwork
def _get_load_balancer_and_network(self):
    try:
        self.hcloud_network = self._client_get_by_name_or_id('networks', self.module.params.get('network'))
        self.hcloud_load_balancer = self._client_get_by_name_or_id('load_balancers', self.module.params.get('load_balancer'))
        self.hcloud_load_balancer_network = None
    except HCloudException as exception:
        self.fail_json_hcloud(exception)