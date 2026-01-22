from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
def addrobj_exists(self):
    cmd = [self.module.get_bin_path('ipadm')]
    cmd.append('show-addr')
    cmd.append(self.addrobj)
    rc, dummy, dummy = self.module.run_command(cmd)
    if rc == 0:
        return True
    else:
        return False