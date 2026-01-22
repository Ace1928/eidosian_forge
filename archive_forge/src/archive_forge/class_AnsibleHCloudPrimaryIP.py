from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
class AnsibleHCloudPrimaryIP(AnsibleHCloud):
    represent = 'hcloud_primary_ip'
    hcloud_primary_ip: BoundPrimaryIP | None = None

    def _prepare_result(self):
        return {'id': to_native(self.hcloud_primary_ip.id), 'name': to_native(self.hcloud_primary_ip.name), 'ip': to_native(self.hcloud_primary_ip.ip), 'type': to_native(self.hcloud_primary_ip.type), 'datacenter': to_native(self.hcloud_primary_ip.datacenter.name), 'labels': self.hcloud_primary_ip.labels, 'delete_protection': self.hcloud_primary_ip.protection['delete']}

    def _get_primary_ip(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_primary_ip = self.client.primary_ips.get_by_id(self.module.params.get('id'))
            else:
                self.hcloud_primary_ip = self.client.primary_ips.get_by_name(self.module.params.get('name'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

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

    def _update_primary_ip(self):
        try:
            labels = self.module.params.get('labels')
            if labels is not None and labels != self.hcloud_primary_ip.labels:
                if not self.module.check_mode:
                    self.hcloud_primary_ip.update(labels=labels)
                self._mark_as_changed()
            delete_protection = self.module.params.get('delete_protection')
            if delete_protection is not None and delete_protection != self.hcloud_primary_ip.protection['delete']:
                if not self.module.check_mode:
                    self.hcloud_primary_ip.change_protection(delete=delete_protection).wait_until_finished()
                self._mark_as_changed()
            self._get_primary_ip()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def present_primary_ip(self):
        self._get_primary_ip()
        if self.hcloud_primary_ip is None:
            self._create_primary_ip()
        else:
            self._update_primary_ip()

    def delete_primary_ip(self):
        try:
            self._get_primary_ip()
            if self.hcloud_primary_ip is not None:
                if not self.module.check_mode:
                    self.client.primary_ips.delete(self.hcloud_primary_ip)
                self._mark_as_changed()
            self.hcloud_primary_ip = None
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, datacenter={'type': 'str'}, auto_delete={'type': 'bool', 'default': False}, type={'choices': ['ipv4', 'ipv6']}, labels={'type': 'dict'}, delete_protection={'type': 'bool'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), required_one_of=[['id', 'name']], supports_check_mode=True)