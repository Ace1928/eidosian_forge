from __future__ import absolute_import, division, print_function
import ctypes.util
import grp
import calendar
import os
import re
import pty
import pwd
import select
import shutil
import socket
import subprocess
import time
import math
from ansible.module_utils import distro
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
import ansible.module_utils.compat.typing as t
def __modify_group(self, group, action):
    """Add or remove SELF.NAME to or from GROUP depending on ACTION.
        ACTION can be 'add' or 'remove' otherwise 'remove' is assumed. """
    if action == 'add':
        option = '-a'
    else:
        option = '-d'
    cmd = ['dseditgroup', '-o', 'edit', option, self.name, '-t', 'user', group]
    rc, out, err = self.execute_command(cmd)
    if rc != 0:
        self.module.fail_json(msg='Cannot %s user "%s" to group "%s".' % (action, self.name, group), err=err, out=out, rc=rc)
    return (rc, out, err)