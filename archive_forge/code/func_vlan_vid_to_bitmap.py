from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def vlan_vid_to_bitmap(vid):
    """convert VLAN list to VLAN bitmap"""
    vlan_bit = ['0'] * 1024
    int_vid = int(vid)
    j = int_vid // 4
    bit_int = 8 >> int_vid % 4
    vlan_bit[j] = str(hex(bit_int))[2]
    return ''.join(vlan_bit)