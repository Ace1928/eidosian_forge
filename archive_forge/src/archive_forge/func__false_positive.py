from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _false_positive(self, devices, have_devices):
    match = 0
    for w in devices:
        for h in have_devices:
            if w.items() == h.items():
                match = match + 1
    if match == len(devices):
        return True