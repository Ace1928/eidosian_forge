from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_traps_vrrp(config_data):
    command = 'snmp-server enable traps vrrp'
    el = config_data['traps']['vrrp']
    if el.get('trap_new_master'):
        command += ' trap-new-master'
    return command