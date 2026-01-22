from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_snmp(config_data):
    command = 'snmp-server enable traps snmp'
    el = config_data['traps']['snmp']
    if el.get('authentication'):
        command += ' authentication'
    if el.get('link_down'):
        command += ' link-down'
    if el.get('link_up'):
        command += ' link-up'
    return command