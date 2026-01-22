from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class VLAN(object):

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.link = module.params['link']
        self.vlan_id = module.params['vlan_id']
        self.temporary = module.params['temporary']
        self.state = module.params['state']

    def vlan_exists(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('show-vlan')
        cmd.append(self.name)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            return False

    def create_vlan(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('create-vlan')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-l')
        cmd.append(self.link)
        cmd.append('-v')
        cmd.append(self.vlan_id)
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def delete_vlan(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('delete-vlan')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def is_valid_vlan_id(self):
        return 0 <= int(self.vlan_id) <= 4095