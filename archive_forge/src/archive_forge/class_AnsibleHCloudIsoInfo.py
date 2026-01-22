from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.isos import BoundIso
class AnsibleHCloudIsoInfo(AnsibleHCloud):
    represent = 'hcloud_iso_info'
    hcloud_iso_info: list[BoundIso] | None = None

    def _prepare_result(self):
        tmp = []
        for iso_info in self.hcloud_iso_info:
            if iso_info is None:
                continue
            tmp.append({'id': to_native(iso_info.id), 'name': to_native(iso_info.name), 'description': to_native(iso_info.description), 'type': iso_info.type, 'architecture': iso_info.architecture, 'deprecated': iso_info.deprecation.unavailable_after if iso_info.deprecation is not None else None, 'deprecation': {'announced': iso_info.deprecation.announced.isoformat(), 'unavailable_after': iso_info.deprecation.unavailable_after.isoformat()} if iso_info.deprecation is not None else None})
        return tmp

    def get_iso_infos(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_iso_info = [self.client.isos.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_iso_info = [self.client.isos.get_by_name(self.module.params.get('name'))]
            else:
                self.hcloud_iso_info = self.client.isos.get_all(architecture=self.module.params.get('architecture'), include_wildcard_architecture=self.module.params.get('include_wildcard_architecture'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, architecture={'type': 'str', 'choices': ['x86', 'arm']}, include_architecture_wildcard={'type': 'bool'}, **super().base_module_arguments()), supports_check_mode=True)