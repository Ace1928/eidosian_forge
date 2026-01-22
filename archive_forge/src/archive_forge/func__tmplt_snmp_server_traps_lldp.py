from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_lldp(config_data):
    command = 'snmp-server enable traps lldp'
    el = config_data['traps']['lldp']
    if el.get('rem_tables_change'):
        command += ' rem-tables-change'
    return command