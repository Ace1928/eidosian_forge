from __future__ import absolute_import, division, print_function
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
def wait_for_removal(self):
    count = 0
    while count < 3:
        if not self.exists():
            count += 1
        else:
            count = 0
        time.sleep(1)