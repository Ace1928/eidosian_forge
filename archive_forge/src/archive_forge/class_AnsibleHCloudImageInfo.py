from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.images import BoundImage
class AnsibleHCloudImageInfo(AnsibleHCloud):
    represent = 'hcloud_image_info'
    hcloud_image_info: list[BoundImage] | None = None

    def _prepare_result(self):
        tmp = []
        for image in self.hcloud_image_info:
            if image is not None:
                tmp.append({'id': to_native(image.id), 'status': to_native(image.status), 'type': to_native(image.type), 'name': to_native(image.name), 'description': to_native(image.description), 'os_flavor': to_native(image.os_flavor), 'os_version': to_native(image.os_version), 'architecture': to_native(image.architecture), 'labels': image.labels})
        return tmp

    def get_images(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_image_info = [self.client.images.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None and self.module.params.get('architecture') is not None:
                self.hcloud_image_info = [self.client.images.get_by_name_and_architecture(self.module.params.get('name'), self.module.params.get('architecture'))]
            elif self.module.params.get('name') is not None:
                self.module.warn('This module only returns x86 images by default. Please set architecture:x86|arm to hide this message.')
                self.hcloud_image_info = [self.client.images.get_by_name(self.module.params.get('name'))]
            else:
                params = {}
                label_selector = self.module.params.get('label_selector')
                if label_selector:
                    params['label_selector'] = label_selector
                image_type = self.module.params.get('type')
                if image_type:
                    params['type'] = image_type
                architecture = self.module.params.get('architecture')
                if architecture:
                    params['architecture'] = architecture
                self.hcloud_image_info = self.client.images.get_all(**params)
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, label_selector={'type': 'str'}, type={'choices': ['system', 'snapshot', 'backup'], 'default': 'system', 'type': 'str'}, architecture={'choices': ['x86', 'arm'], 'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)