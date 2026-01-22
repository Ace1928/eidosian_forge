from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
def _get_load_balancer_service(self):
    for service in self.hcloud_load_balancer.services:
        if self.module.params.get('listen_port') == service.listen_port:
            self.hcloud_load_balancer_service = service