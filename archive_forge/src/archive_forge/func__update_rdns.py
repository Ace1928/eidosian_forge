from __future__ import annotations
import ipaddress
from typing import Any
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.floating_ips import BoundFloatingIP
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
from ..module_utils.vendor.hcloud.servers import BoundServer
def _update_rdns(self):
    dns_ptr = self.module.params.get('dns_ptr')
    if dns_ptr != self.hcloud_rdns['dns_ptr']:
        params = {'ip': self.module.params.get('ip_address'), 'dns_ptr': dns_ptr}
        if not self.module.check_mode:
            try:
                self.hcloud_resource.change_dns_ptr(**params).wait_until_finished()
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_resource()
        self._get_rdns()