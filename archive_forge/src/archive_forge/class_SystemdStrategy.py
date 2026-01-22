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
class SystemdStrategy(BaseStrategy):
    """
    This is a Systemd hostname manipulation strategy class - it uses
    the hostnamectl command.
    """
    COMMAND = 'hostnamectl'

    def __init__(self, module):
        super(SystemdStrategy, self).__init__(module)
        self.hostnamectl_cmd = self.module.get_bin_path(self.COMMAND, True)

    def get_current_hostname(self):
        cmd = [self.hostnamectl_cmd, '--transient', 'status']
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))
        return to_native(out).strip()

    def set_current_hostname(self, name):
        if len(name) > 64:
            self.module.fail_json(msg='name cannot be longer than 64 characters on systemd servers, try a shorter name')
        cmd = [self.hostnamectl_cmd, '--transient', 'set-hostname', name]
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))

    def get_permanent_hostname(self):
        cmd = [self.hostnamectl_cmd, '--static', 'status']
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))
        return to_native(out).strip()

    def set_permanent_hostname(self, name):
        if len(name) > 64:
            self.module.fail_json(msg='name cannot be longer than 64 characters on systemd servers, try a shorter name')
        cmd = [self.hostnamectl_cmd, '--pretty', '--static', 'set-hostname', name]
        rc, out, err = self.module.run_command(cmd)
        if rc != 0:
            self.module.fail_json(msg='Command failed rc=%d, out=%s, err=%s' % (rc, out, err))

    def update_current_and_permanent_hostname(self):
        self.update_permanent_hostname()
        self.update_current_hostname()
        return self.changed