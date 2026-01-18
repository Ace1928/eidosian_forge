from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def user_tmplt(config_data):
    user = config_data.get('user', '')
    group = config_data.get('group', '')
    version = config_data.get('version', '')
    command = 'snmp-server user {user} {group} {version}'.format(user=user, group=group, version=version)
    if config_data.get('acl_v4'):
        command += ' IPv4 {acl_v4}'.format(acl_v4=config_data['acl_v4'])
    if config_data.get('acl_v6'):
        command += ' IPv6 {acl_v6}'.format(acl_v6=config_data['acl_v6'])
    if config_data.get('v4_acl'):
        command += ' {v4_acl}'.format(v4_acl=config_data['v4_acl'])
    if config_data.get('SDROwner'):
        command += ' SDROwner'
    elif config_data.get('SystemOwner'):
        command += ' SystemOwner'
    return command