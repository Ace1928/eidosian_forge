from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork, NetworkRoute
def _create_route(self):
    route = NetworkRoute(destination=self.module.params.get('destination'), gateway=self.module.params.get('gateway'))
    if not self.module.check_mode:
        try:
            self.hcloud_network.add_route(route=route).wait_until_finished()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_network()
    self._get_route()