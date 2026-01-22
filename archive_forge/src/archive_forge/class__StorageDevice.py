import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
class _StorageDevice:

    def __init__(self, image, size):
        self.image = image
        self.size = size

    def to_dict(self):
        extra = self.image.extra
        if extra['type'] == 'template':
            return self._storage_device_for_template_image()
        elif extra['type'] == 'cdrom':
            return self._storage_device_for_cdrom_image()

    def _storage_device_for_template_image(self):
        hdd_device = {'action': 'clone', 'storage': self.image.id}
        hdd_device.update(self._common_hdd_device())
        return {'storage_device': [hdd_device]}

    def _storage_device_for_cdrom_image(self):
        hdd_device = {'action': 'create'}
        hdd_device.update(self._common_hdd_device())
        storage_devices = {'storage_device': [hdd_device, {'action': 'attach', 'storage': self.image.id, 'type': 'cdrom'}]}
        return storage_devices

    def _common_hdd_device(self):
        return {'title': self.image.name, 'size': self.size.disk, 'tier': self.size.extra.get('storage_tier', 'maxiops')}