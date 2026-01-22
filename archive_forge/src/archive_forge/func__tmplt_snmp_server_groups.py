from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_groups(config_data):
    command = 'snmp-server group ' + config_data['groups']['group']
    el = config_data['groups']
    command += ' ' + el['version']
    if el.get('auth_privacy'):
        command += ' ' + el['auth_privacy']
    for param in ['context', 'read', 'write', 'notify']:
        if el.get(param):
            command += ' ' + param + ' ' + el[param]
    return command