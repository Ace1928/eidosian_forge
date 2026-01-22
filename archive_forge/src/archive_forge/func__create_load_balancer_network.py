from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer, PrivateNet
from ..module_utils.vendor.hcloud.networks import BoundNetwork
def _create_load_balancer_network(self):
    params = {'network': self.hcloud_network}
    if self.module.params.get('ip') is not None:
        params['ip'] = self.module.params.get('ip')
    if not self.module.check_mode:
        try:
            self.hcloud_load_balancer.attach_to_network(**params).wait_until_finished()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_load_balancer_and_network()
    self._get_load_balancer_network()