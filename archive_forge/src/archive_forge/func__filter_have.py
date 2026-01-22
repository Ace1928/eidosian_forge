from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _filter_have(self, want, have):
    to_check = set(want.keys()).intersection(set(have.keys()))
    result = dict()
    for k in list(to_check):
        result[k] = have[k]
    return result