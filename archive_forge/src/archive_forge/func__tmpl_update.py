from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_update(config_data):
    update_conf = config_data.get('update', {})
    update_wait = config_data.get('update', {}).get('wait_install')
    update_limit = config_data.get('update', {}).get('limit', {})
    commands = []
    if update_conf:
        if update_wait:
            command = 'update wait-install'
            commands.append(command)
        if 'address_family' in update_limit:
            command = 'update limit address-family ' + str(update_limit['address_family'])
            commands.append(command)
        if 'sub_group' in update_limit:
            if 'ibgp' in update_limit['sub_group']:
                command = 'update limit sub-group ibgp ' + str(update_limit['sub_group']['ibgp'])
                commands.append(command)
            if 'ebgp' in update_limit['sub_group']:
                command = 'update limit sub-group ebgp ' + str(update_limit['sub_group']['ebgp'])
                commands.append(command)
    return commands