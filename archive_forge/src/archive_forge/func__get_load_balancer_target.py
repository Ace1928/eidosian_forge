from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
from ..module_utils.vendor.hcloud.servers import BoundServer
def _get_load_balancer_target(self):
    for target in self.hcloud_load_balancer.targets:
        if self.module.params.get('type') == 'server' and target.type == 'server':
            if target.server.id == self.hcloud_server.id:
                self.hcloud_load_balancer_target = target
        elif self.module.params.get('type') == 'label_selector' and target.type == 'label_selector':
            if target.label_selector.selector == self.module.params.get('label_selector'):
                self.hcloud_load_balancer_target = target
        elif self.module.params.get('type') == 'ip' and target.type == 'ip':
            if target.ip.ip == self.module.params.get('ip'):
                self.hcloud_load_balancer_target = target