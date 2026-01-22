from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_set_variable_action(self, action, item):
    """Handle the nuances of the set_variable type

        :param action:
        :param item:
        :return:
        """
    if 'expression' not in item and 'variable_name' not in item:
        raise F5ModuleError("A 'variable_name' and 'expression' must be specified when the 'set_variable' type is used.")
    if 'event' in item and item['event'] is not None:
        action[item['event']] = True
    else:
        action['request'] = True
    action.update(dict(type='set_variable', expression=item['expression'], tmName=item['variable_name'], setVariable=True, tcl=True))