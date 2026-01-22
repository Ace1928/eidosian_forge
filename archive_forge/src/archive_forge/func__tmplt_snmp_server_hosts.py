from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_hosts(config_data):
    el = list(config_data['hosts'].values())[0]
    command = 'snmp-server host ' + el['host']
    if el.get('vrf'):
        command += ' vrf' + el['vrf']
    if el.get('informs'):
        command += ' informs'
    if el.get('traps'):
        command += ' traps'
    if el.get('version'):
        command += ' version ' + el['version']
    if el.get('user'):
        command += ' ' + el['user']
    if el.get('udp_port'):
        command += ' udp-port ' + str(el['udp_port'])
    return command