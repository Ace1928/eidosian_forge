from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def time_until_up(self):
    if self._values['time_until_up'] is None:
        return None
    return int(self._values['time_until_up'])