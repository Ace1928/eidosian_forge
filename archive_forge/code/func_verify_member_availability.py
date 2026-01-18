from __future__ import absolute_import, division, print_function
import copy
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def verify_member_availability(self):
    if self._values['verify_member_availability'] is None:
        return None
    elif self._values['verify_member_availability']:
        return 'enabled'
    else:
        return 'disabled'