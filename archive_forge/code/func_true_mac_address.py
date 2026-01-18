from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
@property
def true_mac_address(self):
    if self._values['stats']['macTrue'] in [None, 'none']:
        return None
    return self._values['stats']['macTrue']