from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_persist_action(self, action, item):
    """Handle the nuances of the persist type

        :param action:
        :param item:
        :return:
        """
    action['type'] = 'persist'
    if 'cookie_insert' not in item:
        raise F5ModuleError("A 'cookie_insert' must be specified when the 'persist' type is used.")
    elif 'cookie_expiry' in item:
        action.update(cookieInsert=True, tmName=item['cookie_insert'], expiry=str(item['cookie_expiry']))
    else:
        action.update(cookieInsert=True, tmName=item['cookie_insert'])