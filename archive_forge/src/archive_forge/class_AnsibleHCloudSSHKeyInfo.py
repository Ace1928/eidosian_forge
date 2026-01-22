from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.ssh_keys import BoundSSHKey
class AnsibleHCloudSSHKeyInfo(AnsibleHCloud):
    represent = 'hcloud_ssh_key_info'
    hcloud_ssh_key_info: list[BoundSSHKey] | None = None

    def _prepare_result(self):
        ssh_keys = []
        for ssh_key in self.hcloud_ssh_key_info:
            if ssh_key:
                ssh_keys.append({'id': to_native(ssh_key.id), 'name': to_native(ssh_key.name), 'fingerprint': to_native(ssh_key.fingerprint), 'public_key': to_native(ssh_key.public_key), 'labels': ssh_key.labels})
        return ssh_keys

    def get_ssh_keys(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_ssh_key_info = [self.client.ssh_keys.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_ssh_key_info = [self.client.ssh_keys.get_by_name(self.module.params.get('name'))]
            elif self.module.params.get('fingerprint') is not None:
                self.hcloud_ssh_key_info = [self.client.ssh_keys.get_by_fingerprint(self.module.params.get('fingerprint'))]
            elif self.module.params.get('label_selector') is not None:
                self.hcloud_ssh_key_info = self.client.ssh_keys.get_all(label_selector=self.module.params.get('label_selector'))
            else:
                self.hcloud_ssh_key_info = self.client.ssh_keys.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, fingerprint={'type': 'str'}, label_selector={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)