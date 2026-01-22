from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.ssh_keys import BoundSSHKey
class AnsibleHCloudSSHKey(AnsibleHCloud):
    represent = 'hcloud_ssh_key'
    hcloud_ssh_key: BoundSSHKey | None = None

    def _prepare_result(self):
        return {'id': to_native(self.hcloud_ssh_key.id), 'name': to_native(self.hcloud_ssh_key.name), 'fingerprint': to_native(self.hcloud_ssh_key.fingerprint), 'public_key': to_native(self.hcloud_ssh_key.public_key), 'labels': self.hcloud_ssh_key.labels}

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

    def _create_ssh_key(self):
        self.module.fail_on_missing_params(required_params=['name', 'public_key'])
        params = {'name': self.module.params.get('name'), 'public_key': self.module.params.get('public_key'), 'labels': self.module.params.get('labels')}
        if not self.module.check_mode:
            try:
                self.client.ssh_keys.create(**params)
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_ssh_key()

    def _update_ssh_key(self):
        name = self.module.params.get('name')
        if name is not None and self.hcloud_ssh_key.name != name:
            self.module.fail_on_missing_params(required_params=['id'])
            if not self.module.check_mode:
                self.hcloud_ssh_key.update(name=name)
            self._mark_as_changed()
        labels = self.module.params.get('labels')
        if labels is not None and self.hcloud_ssh_key.labels != labels:
            if not self.module.check_mode:
                self.hcloud_ssh_key.update(labels=labels)
            self._mark_as_changed()
        self._get_ssh_key()

    def present_ssh_key(self):
        self._get_ssh_key()
        if self.hcloud_ssh_key is None:
            self._create_ssh_key()
        else:
            self._update_ssh_key()

    def delete_ssh_key(self):
        self._get_ssh_key()
        if self.hcloud_ssh_key is not None:
            if not self.module.check_mode:
                try:
                    self.client.ssh_keys.delete(self.hcloud_ssh_key)
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
            self._mark_as_changed()
        self.hcloud_ssh_key = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, public_key={'type': 'str'}, fingerprint={'type': 'str'}, labels={'type': 'dict'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), required_one_of=[['id', 'name', 'fingerprint']], required_if=[['state', 'present', ['name']]], supports_check_mode=True)