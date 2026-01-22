from __future__ import absolute_import, division, print_function
import grp
import os
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.sys_info import get_platform_subclass
class BusyBoxGroup(Group):
    """
    BusyBox group manipulation class for systems that have addgroup and delgroup.

    It overrides the following methods:
        - group_add()
        - group_del()
        - group_mod()
    """

    def group_add(self, **kwargs):
        cmd = [self.module.get_bin_path('addgroup', True)]
        if self.gid is not None:
            cmd.extend(['-g', str(self.gid)])
        if self.system:
            cmd.append('-S')
        cmd.append(self.name)
        return self.execute_command(cmd)

    def group_del(self):
        cmd = [self.module.get_bin_path('delgroup', True), self.name]
        return self.execute_command(cmd)

    def group_mod(self, **kwargs):
        info = self.group_info()
        if self.gid is not None and self.gid != info[2]:
            with open('/etc/group', 'rb') as f:
                b_groups = f.read()
            b_name = to_bytes(self.name)
            b_current_group_string = b'%s:x:%d:' % (b_name, info[2])
            b_new_group_string = b'%s:x:%d:' % (b_name, self.gid)
            if b':%d:' % self.gid in b_groups:
                self.module.fail_json(msg="gid '{gid}' in use".format(gid=self.gid))
            if self.module.check_mode:
                return (0, '', '')
            b_new_groups = b_groups.replace(b_current_group_string, b_new_group_string)
            with open('/etc/group', 'wb') as f:
                f.write(b_new_groups)
            return (0, '', '')
        return (None, '', '')