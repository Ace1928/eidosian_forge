from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _convert_include_list(self, items):
    if items is None:
        return None
    result = list()
    for item in items:
        element = dict()
        element['url'] = item['url']
        if 'threshold' in item:
            element['threshold'] = item['threshold']
        result.append(element)
    if result:
        return result