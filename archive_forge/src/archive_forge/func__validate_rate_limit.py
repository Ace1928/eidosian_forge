from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_dictionary
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _validate_rate_limit(self, rate_limit):
    if rate_limit is None:
        return None
    if rate_limit == 'indefinite':
        return 4294967295
    if 0 <= int(rate_limit) <= 4294967295:
        return int(rate_limit)
    raise F5ModuleError("Valid 'maximum_age' must be in range 0 - 4294967295, or 'indefinite'.")