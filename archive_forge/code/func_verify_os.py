from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def verify_os(self):
    """Verifies filesystem against specified or current version"""
    if self.module.check_mode:
        self.module.exit_json(changed=self._needs_verify())
    if not self._needs_verify():
        self.msg = 'No files where changed'
        return
    cmd = self._get_cmd('verify --fix')
    self._run_cmd(cmd)
    if self.rc == 0 and (self.FILES_REPLACED in self.stdout or self.FILES_FIXED in self.stdout or self.FILES_DELETED in self.stdout):
        self.changed = True
        self.msg = 'Fix successful'
        return
    self.failed = True
    self.msg = 'Failed to verify the OS'