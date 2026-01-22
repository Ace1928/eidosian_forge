from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
from ..module_utils.vendor.hcloud.servers import BoundServer
def _get_load_balancer_and_target(self):
    try:
        self.hcloud_load_balancer = self._client_get_by_name_or_id('load_balancers', self.module.params.get('load_balancer'))
        if self.module.params.get('type') == 'server':
            self.hcloud_server = self._client_get_by_name_or_id('servers', self.module.params.get('server'))
        self.hcloud_load_balancer_target = None
    except HCloudException as exception:
        self.fail_json_hcloud(exception)