from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def vlan_bitmap_add(self, oldmap, newmap):
    """vlan add bitmap"""
    vlan_bit = ['0'] * 1024
    if len(newmap) != 1024:
        self.module.fail_json(msg='Error: New vlan bitmap is invalid.')
    if len(oldmap) != 1024 and len(oldmap) != 0:
        self.module.fail_json(msg='Error: old vlan bitmap is invalid.')
    if len(oldmap) == 0:
        return newmap
    for num in range(1024):
        new_tmp = int(newmap[num], 16)
        old_tmp = int(oldmap[num], 16)
        add = ~(new_tmp & old_tmp) & new_tmp
        vlan_bit[num] = hex(add)[2]
    vlan_xml = ''.join(vlan_bit)
    return vlan_xml