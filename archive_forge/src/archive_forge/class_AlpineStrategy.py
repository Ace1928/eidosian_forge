from __future__ import absolute_import, division, print_function
import os
import platform
import socket
import traceback
import ansible.module_utils.compat.typing as t
from ansible.module_utils.basic import (
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.facts.system.service_mgr import ServiceMgrFactCollector
from ansible.module_utils.facts.utils import get_file_lines, get_file_content
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import PY3, text_type
class AlpineStrategy(FileStrategy):
    """
    This is a Alpine Linux Hostname manipulation strategy class - it edits
    the /etc/hostname file then run hostname -F /etc/hostname.
    """
    FILE = '/etc/hostname'
    COMMAND = 'hostname'

    def set_current_hostname(self, name):
        super(AlpineStrategy, self).set_current_hostname(name)
        hostname_cmd = self.module.get_bin_path(self.COMMAND, True)
        cmd = [hostname_cmd, '-F', self.FILE]
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))