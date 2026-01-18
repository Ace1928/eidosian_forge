from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def transform_ip_tos(self, key):
    if self._values[key] is None:
        return None
    try:
        result = int(self._values[key])
        return result
    except ValueError:
        if self._values[key] == 'mimic':
            return 65534
        return self._values[key]