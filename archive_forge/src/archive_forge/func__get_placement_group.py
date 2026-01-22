from __future__ import annotations
from datetime import datetime, timedelta, timezone
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.firewalls import FirewallResource
from ..module_utils.vendor.hcloud.servers import (
from ..module_utils.vendor.hcloud.ssh_keys import SSHKey
from ..module_utils.vendor.hcloud.volumes import Volume
def _get_placement_group(self):
    if self.module.params.get('placement_group') is None:
        return None
    placement_group = self.client.placement_groups.get_by_name(self.module.params.get('placement_group'))
    if placement_group is None:
        try:
            placement_group = self.client.placement_groups.get_by_id(self.module.params.get('placement_group'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception, msg=f'placement_group {self.module.params.get('placement_group')} was not found')
    return placement_group