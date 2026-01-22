from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_vlan_ip_range(self):
    ip_range = self.get_vlan_ip_range()
    if ip_range:
        self.result['changed'] = True
        args = {'id': ip_range['id']}
        if not self.module.check_mode:
            self.query_api('deleteVlanIpRange', **args)
    return ip_range