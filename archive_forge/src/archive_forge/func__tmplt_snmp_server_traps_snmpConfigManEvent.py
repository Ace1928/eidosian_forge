from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_snmpConfigManEvent(config_data):
    command = 'snmp-server enable traps snmpConfigManEvent'
    el = config_data['traps']['snmpConfigManEvent']
    if el.get('arista_config_man_event'):
        command += ' arista-config-man-event'
    return command