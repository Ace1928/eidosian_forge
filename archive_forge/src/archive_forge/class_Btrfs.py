from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class Btrfs(Filesystem):
    MKFS = 'mkfs.btrfs'
    INFO = 'btrfs'
    GROW = 'btrfs'
    GROW_MAX_SPACE_FLAGS = ['filesystem', 'resize', 'max']
    GROW_MOUNTPOINT_ONLY = True

    def __init__(self, module):
        super(Btrfs, self).__init__(module)
        mkfs = self.module.get_bin_path(self.MKFS, required=True)
        dummy, stdout, stderr = self.module.run_command([mkfs, '--version'], check_rc=True)
        match = re.search(' v([0-9.]+)', stdout)
        if not match:
            match = re.search(' v([0-9.]+)', stderr)
        if match:
            if LooseVersion(match.group(1)) >= LooseVersion('3.12'):
                self.MKFS_FORCE_FLAGS = ['-f']
        else:
            self.MKFS_FORCE_FLAGS = ['-f']
            self.module.warn('Unable to identify mkfs.btrfs version (%r, %r)' % (stdout, stderr))

    def get_fs_size(self, dev):
        """Return size in bytes of filesystem on device (integer)."""
        mountpoint = dev.get_mountpoint()
        if not mountpoint:
            self.module.fail_json(msg='%s needs to be mounted for %s operations' % (dev, self.fstype))
        dummy, stdout, dummy = self.module.run_command([self.module.get_bin_path(self.INFO), 'filesystem', 'usage', '-b', mountpoint], check_rc=True)
        for line in stdout.splitlines():
            if 'Device size' in line:
                return int(line.split()[-1])
        raise ValueError(repr(stdout))