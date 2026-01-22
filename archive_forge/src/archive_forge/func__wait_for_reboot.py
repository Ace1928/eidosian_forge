from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _wait_for_reboot(self):
    nops = 0
    last_reboot = self._get_last_reboot()
    time.sleep(5)
    while nops < 6:
        try:
            self.client.reconnect()
            next_reboot = self._get_last_reboot()
            if next_reboot is None:
                nops = 0
            if next_reboot == last_reboot:
                nops = 0
            else:
                nops += 1
        except Exception:
            pass
        time.sleep(10)