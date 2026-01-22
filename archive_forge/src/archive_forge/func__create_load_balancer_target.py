from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
from ..module_utils.vendor.hcloud.servers import BoundServer
def _create_load_balancer_target(self):
    params = {'target': None}
    if self.module.params.get('type') == 'server':
        self.module.fail_on_missing_params(required_params=['server'])
        params['target'] = LoadBalancerTarget(type=self.module.params.get('type'), server=self.hcloud_server, use_private_ip=self.module.params.get('use_private_ip'))
    elif self.module.params.get('type') == 'label_selector':
        self.module.fail_on_missing_params(required_params=['label_selector'])
        params['target'] = LoadBalancerTarget(type=self.module.params.get('type'), label_selector=LoadBalancerTargetLabelSelector(selector=self.module.params.get('label_selector')), use_private_ip=self.module.params.get('use_private_ip'))
    elif self.module.params.get('type') == 'ip':
        self.module.fail_on_missing_params(required_params=['ip'])
        params['target'] = LoadBalancerTarget(type=self.module.params.get('type'), ip=LoadBalancerTargetIP(ip=self.module.params.get('ip')), use_private_ip=False)
    if not self.module.check_mode:
        try:
            self.hcloud_load_balancer.add_target(**params).wait_until_finished()
        except APIException as exception:
            if exception.code == 'locked' or exception.code == 'conflict':
                self._create_load_balancer_target()
            else:
                self.fail_json_hcloud(exception)
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_load_balancer_and_target()
    self._get_load_balancer_target()