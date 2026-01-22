from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _wait_for_fqdn_checks(self):
    while True:
        have = self.read_current_from_device()
        if have.state == 'fqdn-checking':
            time.sleep(1)
        else:
            break