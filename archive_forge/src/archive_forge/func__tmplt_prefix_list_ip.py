from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_prefix_list_ip(config_data):
    command_set = []
    config_data = config_data['prefix_lists'].get('entries', {})
    for k, v in iteritems(config_data):
        command = ''
        if k != 'seq':
            command = 'seq ' + str(k) + ' {action} {address}'.format(**v)
        else:
            command = '{action} {address}'.format(**v)
        if 'match' in v:
            command += ' {operator} {masklen}'.format(**v['match'])
        if command:
            command_set.append(command)
    return command_set