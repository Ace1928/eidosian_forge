from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class IPInterface(object):

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.temporary = module.params['temporary']
        self.state = module.params['state']

    def interface_exists(self):
        cmd = [self.module.get_bin_path('ipadm', True)]
        cmd.append('show-if')
        cmd.append(self.name)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            return False

    def interface_is_disabled(self):
        cmd = [self.module.get_bin_path('ipadm', True)]
        cmd.append('show-if')
        cmd.append('-o')
        cmd.append('state')
        cmd.append(self.name)
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(name=self.name, rc=rc, msg=err)
        return 'disabled' in out

    def create_interface(self):
        cmd = [self.module.get_bin_path('ipadm', True)]
        cmd.append('create-if')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def delete_interface(self):
        cmd = [self.module.get_bin_path('ipadm', True)]
        cmd.append('delete-if')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def enable_interface(self):
        cmd = [self.module.get_bin_path('ipadm', True)]
        cmd.append('enable-if')
        cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def disable_interface(self):
        cmd = [self.module.get_bin_path('ipadm', True)]
        cmd.append('disable-if')
        cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)