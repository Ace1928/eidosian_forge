from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork
from ..module_utils.vendor.hcloud.servers import BoundServer, PrivateNet
def _create_server_network(self):
    params = {'network': self.hcloud_network}
    if self.module.params.get('ip') is not None:
        params['ip'] = self.module.params.get('ip')
    if self.module.params.get('alias_ips') is not None:
        params['alias_ips'] = self.module.params.get('alias_ips')
    if not self.module.check_mode:
        try:
            self.hcloud_server.attach_to_network(**params).wait_until_finished()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_server_and_network()
    self._get_server_network()