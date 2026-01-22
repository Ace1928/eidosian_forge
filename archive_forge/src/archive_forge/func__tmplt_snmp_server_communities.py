from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_communities(config_data):
    config_data = config_data['communities']
    command = []
    cmd = 'service snmp community {name}'.format(**config_data)
    if 'authorization_type' in config_data:
        auth_cmd = cmd + ' authorization {authorization_type}'.format(**config_data)
        command.append(auth_cmd)
    if 'clients' in config_data:
        for c in config_data['clients']:
            client_cmd = cmd + ' client ' + c
            command.append(client_cmd)
    if 'networks' in config_data:
        for n in config_data['networks']:
            network_command = cmd + ' network ' + n
            command.append(network_command)
    if not command:
        command.append(cmd)
    return command