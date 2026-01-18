from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def traffic_group(self):
    if self._values['traffic_group'] is None:
        return None
    if self._values['traffic_group'] == '':
        return ''
    result = fq_name('Common', self._values['traffic_group'])
    return result