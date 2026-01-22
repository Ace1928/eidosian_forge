from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.floating_ips import BoundFloatingIP
def _create_floating_ip(self):
    self.module.fail_on_missing_params(required_params=['type'])
    try:
        params = {'description': self.module.params.get('description'), 'type': self.module.params.get('type'), 'name': self.module.params.get('name')}
        if self.module.params.get('home_location') is not None:
            params['home_location'] = self.client.locations.get_by_name(self.module.params.get('home_location'))
        elif self.module.params.get('server') is not None:
            params['server'] = self.client.servers.get_by_name(self.module.params.get('server'))
        else:
            self.module.fail_json(msg='one of the following is required: home_location, server')
        if self.module.params.get('labels') is not None:
            params['labels'] = self.module.params.get('labels')
        if not self.module.check_mode:
            resp = self.client.floating_ips.create(**params)
            self.hcloud_floating_ip = resp.floating_ip
            delete_protection = self.module.params.get('delete_protection')
            if delete_protection is not None:
                self.hcloud_floating_ip.change_protection(delete=delete_protection).wait_until_finished()
    except HCloudException as exception:
        self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_floating_ip()