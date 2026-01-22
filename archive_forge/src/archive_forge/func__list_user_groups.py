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
def _list_user_groups(self):
    cmd = self._get_dscl()
    cmd += ['-search', '/Groups', 'GroupMembership', self.name]
    rc, out, err = self.execute_command(cmd, obey_checkmode=False)
    groups = []
    for line in out.splitlines():
        if line.startswith(' ') or line.startswith(')'):
            continue
        groups.append(line.split()[0])
    return groups