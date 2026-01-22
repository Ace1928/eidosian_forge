from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class IfProp(object):

    def __init__(self, module):
        self.module = module
        self.interface = module.params['interface']
        self.protocol = module.params['protocol']
        self.property = module.params['property']
        self.value = module.params['value']
        self.temporary = module.params['temporary']
        self.state = module.params['state']

    def property_exists(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('show-ifprop')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append('-m')
        cmd.append(self.protocol)
        cmd.append(self.interface)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            self.module.fail_json(msg='Unknown %s property "%s" on IP interface %s' % (self.protocol, self.property, self.interface), protocol=self.protocol, property=self.property, interface=self.interface)

    def property_is_modified(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('show-ifprop')
        cmd.append('-c')
        cmd.append('-o')
        cmd.append('current,default')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append('-m')
        cmd.append(self.protocol)
        cmd.append(self.interface)
        rc, out, dummy = self.module.run_command(cmd)
        out = out.rstrip()
        value, default = out.split(':')
        if rc == 0 and value == default:
            return True
        else:
            return False

    def property_is_set(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('show-ifprop')
        cmd.append('-c')
        cmd.append('-o')
        cmd.append('current')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append('-m')
        cmd.append(self.protocol)
        cmd.append(self.interface)
        rc, out, dummy = self.module.run_command(cmd)
        out = out.rstrip()
        if rc == 0 and self.value == out:
            return True
        else:
            return False

    def set_property(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('set-ifprop')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-p')
        cmd.append(self.property + '=' + self.value)
        cmd.append('-m')
        cmd.append(self.protocol)
        cmd.append(self.interface)
        return self.module.run_command(cmd)

    def reset_property(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('reset-ifprop')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append('-m')
        cmd.append(self.protocol)
        cmd.append(self.interface)
        return self.module.run_command(cmd)