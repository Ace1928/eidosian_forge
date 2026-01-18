from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def wait_for_provisioning(self):
    delay, period = self.want.status_timeout
    checks = 0
    for x in range(0, period):
        if not self.device_is_ready():
            self.changes.update({'message': 'Device is restarting services, unable to check provisioning status.'})
            return False
        if not self._is_mprov_running_on_device():
            checks += 1
        if checks > 2:
            if self.want.state == 'absent':
                self.changes.update({'message': 'Device has finished de-provisioning the requested module.'})
            else:
                self.changes.update({'message': 'Device has finished provisioning the requested module.'})
            return True
        time.sleep(delay)
    raise F5ModuleError('Module timeout reached, state change is unknown, please increase the status_timeout parameter for long lived actions.')