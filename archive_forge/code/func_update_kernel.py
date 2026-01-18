from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from ansible.module_utils.basic import (
from ansible.module_utils.common.text.converters import to_native
def update_kernel(module):
    rc, out, err = module.run_command(['/usr/sbin/update-kernel', '-y'], check_rc=True, environ_update={'LANG': 'C'})
    return (UPDATE_KERNEL_ZERO not in out, out)