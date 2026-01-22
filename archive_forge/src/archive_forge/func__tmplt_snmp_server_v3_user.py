from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_snmp_server_v3_user(config_data):
    config_data = config_data['snmp_v3']['users']
    command = []
    cmd = 'service snmp v3 user {user}'.format(**config_data)
    for k in ['authentication', 'privacy']:
        if config_data.get(k):
            config = config_data[k]
            if k == 'authentication':
                val = ' auth'
            else:
                val = ' privacy'
            if 'type' in config:
                type_cmd = cmd + val + ' type {type}'.format(**config)
                command.append(type_cmd)
            if 'encrypted_key' in config:
                enc_cmd = cmd + val + ' encrypted-key {encrypted_key}'.format(**config)
                command.append(enc_cmd)
            if 'plaintext_key' in config:
                plain_cmd = cmd + val + ' plaintext-key {plaintext_key}'.format(**config)
                command.append(plain_cmd)
    return command