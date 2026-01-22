from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _validate_threshold(self, item):
    if item == 'auto':
        return item
    if 1 <= int(item) <= 4294967295:
        return item
    raise F5ModuleError("Valid 'url threshold' must be in range 1 - 4294967295 requests per second or 'auto'.")