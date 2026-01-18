from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def wipefs(self, dev):
    if self.module.check_mode:
        return
    wipefs = self.module.get_bin_path('wipefs', required=True)
    cmd = [wipefs, '--all', str(dev)]
    self.module.run_command(cmd, check_rc=True)