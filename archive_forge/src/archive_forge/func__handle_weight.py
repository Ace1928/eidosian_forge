from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_weight(self, weight):
    if weight < 10 or weight > 100:
        raise F5ModuleError("Weight value must be in the range: '10 - 100'.")
    return weight