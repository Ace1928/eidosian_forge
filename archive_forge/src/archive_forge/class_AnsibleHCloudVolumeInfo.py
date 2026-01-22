from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.volumes import BoundVolume
class AnsibleHCloudVolumeInfo(AnsibleHCloud):
    represent = 'hcloud_volume_info'
    hcloud_volume_info: list[BoundVolume] | None = None

    def _prepare_result(self):
        tmp = []
        for volume in self.hcloud_volume_info:
            if volume is not None:
                server_name = None
                if volume.server is not None:
                    server_name = to_native(volume.server.name)
                tmp.append({'id': to_native(volume.id), 'name': to_native(volume.name), 'size': volume.size, 'location': to_native(volume.location.name), 'labels': volume.labels, 'server': server_name, 'linux_device': to_native(volume.linux_device), 'delete_protection': volume.protection['delete']})
        return tmp

    def get_volumes(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_volume_info = [self.client.volumes.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_volume_info = [self.client.volumes.get_by_name(self.module.params.get('name'))]
            elif self.module.params.get('label_selector') is not None:
                self.hcloud_volume_info = self.client.volumes.get_all(label_selector=self.module.params.get('label_selector'))
            else:
                self.hcloud_volume_info = self.client.volumes.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, label_selector={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)