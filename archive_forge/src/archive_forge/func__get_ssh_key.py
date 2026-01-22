from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.ssh_keys import BoundSSHKey
def _get_ssh_key(self):
    try:
        if self.module.params.get('id') is not None:
            self.hcloud_ssh_key = self.client.ssh_keys.get_by_id(self.module.params.get('id'))
        elif self.module.params.get('fingerprint') is not None:
            self.hcloud_ssh_key = self.client.ssh_keys.get_by_fingerprint(self.module.params.get('fingerprint'))
        elif self.module.params.get('name') is not None:
            self.hcloud_ssh_key = self.client.ssh_keys.get_by_name(self.module.params.get('name'))
    except HCloudException as exception:
        self.fail_json_hcloud(exception)