import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
def _storage_device_for_template_image(self):
    hdd_device = {'action': 'clone', 'storage': self.image.id}
    hdd_device.update(self._common_hdd_device())
    return {'storage_device': [hdd_device]}