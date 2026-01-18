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
def wait_for_software_install_on_device(self):
    for dummy in range(10):
        try:
            if self.volume_exists():
                break
        except F5ModuleError:
            pass
        time.sleep(5)
    while True:
        time.sleep(10)
        volume = self.read_volume_from_device()
        if volume is None or 'status' not in volume:
            self.client.reconnect()
            continue
        if volume['status'] == 'complete':
            break
        elif volume['status'] == 'failed':
            raise F5ModuleError