from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
class VNIC(object):
    UNICAST_MAC_REGEX = '^[a-f0-9][2-9a-f0]:([a-f0-9]{2}:){4}[a-f0-9]{2}$'

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.link = module.params['link']
        self.mac = module.params['mac']
        self.vlan = module.params['vlan']
        self.temporary = module.params['temporary']
        self.state = module.params['state']

    def vnic_exists(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('show-vnic')
        cmd.append(self.name)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            return False

    def create_vnic(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('create-vnic')
        if self.temporary:
            cmd.append('-t')
        if self.mac:
            cmd.append('-m')
            cmd.append(self.mac)
        if self.vlan:
            cmd.append('-v')
            cmd.append(self.vlan)
        cmd.append('-l')
        cmd.append(self.link)
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def delete_vnic(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('delete-vnic')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def is_valid_unicast_mac(self):
        mac_re = re.match(self.UNICAST_MAC_REGEX, self.mac)
        return mac_re is not None

    def is_valid_vlan_id(self):
        return 0 < self.vlan < 4095