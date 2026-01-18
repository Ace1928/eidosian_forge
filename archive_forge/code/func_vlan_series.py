from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def vlan_series(self, vlanid_s):
    """ convert vlan range to vlan list """
    vlan_list = []
    peerlistlen = len(vlanid_s)
    if peerlistlen != 2:
        self.module.fail_json(msg='Error: Format of vlanid is invalid.')
    for num in range(peerlistlen):
        if not vlanid_s[num].isdigit():
            self.module.fail_json(msg='Error: Format of vlanid is invalid.')
    if int(vlanid_s[0]) > int(vlanid_s[1]):
        self.module.fail_json(msg='Error: Format of vlanid is invalid.')
    elif int(vlanid_s[0]) == int(vlanid_s[1]):
        vlan_list.append(str(vlanid_s[0]))
        return vlan_list
    for num in range(int(vlanid_s[0]), int(vlanid_s[1])):
        vlan_list.append(str(num))
    vlan_list.append(vlanid_s[1])
    return vlan_list