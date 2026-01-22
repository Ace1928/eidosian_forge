from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.utils.utils import (
def _tmplt_ntp_global_authentication_keys(config_data):
    el = config_data['authentication_keys']
    command = 'ntp authentication-key '
    command += str(el['id'])
    command += ' ' + el['algorithm']
    if 'encryption' in el:
        command += ' ' + str(el['encryption'])
    if 'key' in el:
        command += ' ' + el['key']
    return command