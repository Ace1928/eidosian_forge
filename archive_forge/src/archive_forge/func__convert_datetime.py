from __future__ import absolute_import, division, print_function
import re
import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _convert_datetime(self, value):
    if value is None:
        return None
    p = '(\\d{4})-(\\d{1,2})-(\\d{1,2})[:, T](\\d{2}):(\\d{2}):(\\d{2})'
    match = re.match(p, value)
    if match:
        date = '{0}-{1}-{2}:{3}:{4}:{5}'.format(*match.group(1, 2, 3, 4, 5, 6))
        return date