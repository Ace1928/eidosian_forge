from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _validate_aggregate_rate(self, aggregate_rate):
    if aggregate_rate is None:
        return None
    if aggregate_rate == 'indefinite':
        return 4294967295
    if 0 <= int(aggregate_rate) <= 4294967295:
        return int(aggregate_rate)
    raise F5ModuleError("Valid 'maximum_age' must be in range 0 - 4294967295, or 'indefinite'.")