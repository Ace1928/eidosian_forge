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
def _check_usermod_append(self):
    if self.local:
        command_name = 'lusermod'
    else:
        command_name = 'usermod'
    usermod_path = self.module.get_bin_path(command_name, True)
    if not os.access(usermod_path, os.X_OK):
        return False
    cmd = [usermod_path, '--help']
    rc, data1, data2 = self.execute_command(cmd, obey_checkmode=False)
    helpout = data1 + data2
    lines = to_native(helpout).split('\n')
    for line in lines:
        if line.strip().startswith('-a, --append'):
            return True
    return False