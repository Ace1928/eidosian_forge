from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_address_family_cmd(config_data):
    if 'address_family' in config_data:
        command = ['address-family ipv4 multicast', 'exit-address-family']
        if config_data['address_family'].get('topology'):
            if 'base' in config_data['address_family'].get('topology'):
                command.insert(1, 'topology base')
            elif 'name' in config_data['address_family'].get('topology'):
                cmd = 'topology {name}'.format(**config_data['address_family'].get('topology'))
                if 'tid' in config_data['address_family'].get('topology'):
                    cmd += ' tid {tid}'.format(**config_data['address_family'].get('topology'))
                command.insert(1, cmd)
        return command