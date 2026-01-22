from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_capacity(config_data):
    command = 'snmp-server enable traps capacity'
    el = config_data['traps']['capacity']
    if el.get('arista_hardware_utilization_alert'):
        command += ' arista-hardware-utilization-alert'
    return command