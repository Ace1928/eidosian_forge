from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_switchover(config_data):
    command = 'snmp-server enable traps switchover'
    el = config_data['traps']['switchover']
    if el.get('arista_redundancy_switch_over_notif'):
        command += ' arista-redundancy-switch-over-notif'
    return command