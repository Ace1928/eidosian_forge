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
class RedHatStrategy(BaseStrategy):
    """
    This is a Redhat Hostname strategy class - it edits the
    /etc/sysconfig/network file.
    """
    NETWORK_FILE = '/etc/sysconfig/network'

    def get_permanent_hostname(self):
        try:
            for line in get_file_lines(self.NETWORK_FILE):
                line = to_native(line).strip()
                if line.startswith('HOSTNAME'):
                    k, v = line.split('=')
                    return v.strip()
            self.module.fail_json('Unable to locate HOSTNAME entry in %s' % self.NETWORK_FILE)
        except Exception as e:
            self.module.fail_json(msg='failed to read hostname: %s' % to_native(e), exception=traceback.format_exc())

    def set_permanent_hostname(self, name):
        try:
            lines = []
            found = False
            content = get_file_content(self.NETWORK_FILE, strip=False) or ''
            for line in content.splitlines(True):
                line = to_native(line)
                if line.strip().startswith('HOSTNAME'):
                    lines.append('HOSTNAME=%s\n' % name)
                    found = True
                else:
                    lines.append(line)
            if not found:
                lines.append('HOSTNAME=%s\n' % name)
            with open(self.NETWORK_FILE, 'w+') as f:
                f.writelines(lines)
        except Exception as e:
            self.module.fail_json(msg='failed to update hostname: %s' % to_native(e), exception=traceback.format_exc())