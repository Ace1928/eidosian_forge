from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
def add_nictag(self):
    cmd = [self.nictagadm_bin, '-v', 'add']
    if self.etherstub:
        cmd.append('-l')
    if self.mtu:
        cmd.append('-p')
        cmd.append('mtu=' + str(self.mtu))
    if self.mac:
        cmd.append('-p')
        cmd.append('mac=' + str(self.mac))
    cmd.append(self.name)
    return self.module.run_command(cmd)