from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class LinkProp(object):

    def __init__(self, module):
        self.module = module
        self.link = module.params['link']
        self.property = module.params['property']
        self.value = module.params['value']
        self.temporary = module.params['temporary']
        self.state = module.params['state']
        self.dladm_bin = self.module.get_bin_path('dladm', True)

    def property_exists(self):
        cmd = [self.dladm_bin]
        cmd.append('show-linkprop')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.link)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            self.module.fail_json(msg='Unknown property "%s" on link %s' % (self.property, self.link), property=self.property, link=self.link)

    def property_is_modified(self):
        cmd = [self.dladm_bin]
        cmd.append('show-linkprop')
        cmd.append('-c')
        cmd.append('-o')
        cmd.append('value,default')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.link)
        rc, out, dummy = self.module.run_command(cmd)
        out = out.rstrip()
        value, default = out.split(':')
        if rc == 0 and value == default:
            return True
        else:
            return False

    def property_is_readonly(self):
        cmd = [self.dladm_bin]
        cmd.append('show-linkprop')
        cmd.append('-c')
        cmd.append('-o')
        cmd.append('perm')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.link)
        rc, out, dummy = self.module.run_command(cmd)
        out = out.rstrip()
        if rc == 0 and out == 'r-':
            return True
        else:
            return False

    def property_is_set(self):
        cmd = [self.dladm_bin]
        cmd.append('show-linkprop')
        cmd.append('-c')
        cmd.append('-o')
        cmd.append('value')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.link)
        rc, out, dummy = self.module.run_command(cmd)
        out = out.rstrip()
        if rc == 0 and self.value == out:
            return True
        else:
            return False

    def set_property(self):
        cmd = [self.dladm_bin]
        cmd.append('set-linkprop')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-p')
        cmd.append(self.property + '=' + self.value)
        cmd.append(self.link)
        return self.module.run_command(cmd)

    def reset_property(self):
        cmd = [self.dladm_bin]
        cmd.append('reset-linkprop')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.link)
        return self.module.run_command(cmd)