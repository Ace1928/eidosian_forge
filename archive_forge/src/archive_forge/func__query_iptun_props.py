from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _query_iptun_props(self):
    cmd = [self.dladm_bin]
    cmd.append('show-iptun')
    cmd.append('-p')
    cmd.append('-c')
    cmd.append('link,type,flags,local,remote')
    cmd.append(self.name)
    return self.module.run_command(cmd)