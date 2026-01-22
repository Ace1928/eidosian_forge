from __future__ import absolute_import, division, print_function
import os
import re
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _is_config_reloading_failed_on_device(self, output):
    failed = 'Last Configuration Load Status\\s+base-config-load-failed'
    matches = re.search(failed, output)
    if matches:
        return True
    return False