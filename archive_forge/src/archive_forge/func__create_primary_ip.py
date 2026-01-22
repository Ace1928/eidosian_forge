from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
def _create_primary_ip(self):
    self.module.fail_on_missing_params(required_params=['type', 'datacenter'])
    try:
        params = {'type': self.module.params.get('type'), 'name': self.module.params.get('name'), 'datacenter': self.client.datacenters.get_by_name(self.module.params.get('datacenter'))}
        if self.module.params.get('labels') is not None:
            params['labels'] = self.module.params.get('labels')
        if not self.module.check_mode:
            resp = self.client.primary_ips.create(**params)
            self.hcloud_primary_ip = resp.primary_ip
            delete_protection = self.module.params.get('delete_protection')
            if delete_protection is not None:
                self.hcloud_primary_ip.change_protection(delete=delete_protection).wait_until_finished()
    except HCloudException as exception:
        self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_primary_ip()