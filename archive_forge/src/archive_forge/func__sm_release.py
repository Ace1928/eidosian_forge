from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def _sm_release(module, *args):
    sm_bin = module.get_bin_path('subscription-manager', required=True)
    cmd = [sm_bin, 'release'] + list(args)
    return module.run_command(cmd, check_rc=True, expand_user_and_vars=False)