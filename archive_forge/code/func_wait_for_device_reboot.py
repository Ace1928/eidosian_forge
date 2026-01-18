from __future__ import absolute_import, division, print_function
import time
import ssl
from datetime import datetime
from ansible.module_utils.six.moves.urllib.error import URLError
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def wait_for_device_reboot(self):
    while True:
        time.sleep(5)
        try:
            self.client.reconnect()
            volume = self.read_volume_from_device()
            if volume is None:
                continue
            if 'active' in volume and volume['active'] is True:
                break
        except F5ModuleError:
            pass