from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def upgrade_xbps(module, xbps_path, exit_on_success=False):
    cmdupgradexbps = '%s -uy xbps' % xbps_path['install']
    rc, stdout, stderr = module.run_command(cmdupgradexbps, check_rc=False)
    if rc != 0:
        module.fail_json(msg='Could not upgrade xbps itself')