from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.volumes import BoundVolume
def _create_volume(self):
    self.module.fail_on_missing_params(required_params=['name', 'size'])
    params = {'name': self.module.params.get('name'), 'size': self.module.params.get('size'), 'automount': self.module.params.get('automount'), 'format': self.module.params.get('format'), 'labels': self.module.params.get('labels')}
    if self.module.params.get('server') is not None:
        params['server'] = self.client.servers.get_by_name(self.module.params.get('server'))
    elif self.module.params.get('location') is not None:
        params['location'] = self.client.locations.get_by_name(self.module.params.get('location'))
    else:
        self.module.fail_json(msg='server or location is required')
    if not self.module.check_mode:
        try:
            resp = self.client.volumes.create(**params)
            resp.action.wait_until_finished()
            [action.wait_until_finished() for action in resp.next_actions]
            delete_protection = self.module.params.get('delete_protection')
            if delete_protection is not None:
                self._get_volume()
                self.hcloud_volume.change_protection(delete=delete_protection).wait_until_finished()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_volume()