from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def _get_ssh_version(self):
    ssh_bin = self.module.get_bin_path('ssh')
    if not ssh_bin:
        return ''
    return parse_openssh_version(self.module.run_command([ssh_bin, '-V', '-q'], check_rc=True)[2].strip())