from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_v3_groups(config_data):
    config_data = config_data['snmp_v3']['groups']
    command = []
    cmd = 'service snmp v3 group {group}'.format(**config_data)
    if 'mode' in config_data:
        mode_cmd = cmd + ' mode {mode}'.format(**config_data)
        command.append(mode_cmd)
    if 'seclevel' in config_data:
        sec_cmd = cmd + ' seclevel {seclevel}'.format(**config_data)
        command.append(sec_cmd)
    if 'view' in config_data:
        view_cmd = cmd + ' view {view}'.format(**config_data)
        command.append(view_cmd)
    return command