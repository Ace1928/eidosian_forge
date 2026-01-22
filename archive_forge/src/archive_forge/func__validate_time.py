from __future__ import absolute_import, division, print_function
import re
import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _validate_time(self, value):
    p = '(\\d{2}):(\\d{2})'
    match = re.match(p, value)
    if match:
        time = (int(match.group(1)), int(match.group(2)))
        try:
            datetime.time(*time)
        except ValueError as ex:
            raise F5ModuleError(str(ex))