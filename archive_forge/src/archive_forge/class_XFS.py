from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class XFS(Filesystem):
    MKFS = 'mkfs.xfs'
    MKFS_FORCE_FLAGS = ['-f']
    INFO = 'xfs_info'
    GROW = 'xfs_growfs'
    GROW_MOUNTPOINT_ONLY = True
    CHANGE_UUID = 'xfs_admin'
    CHANGE_UUID_OPTION = '-U'

    def get_fs_size(self, dev):
        """Get bsize and blocks and return their product."""
        cmdline = [self.module.get_bin_path(self.INFO, required=True)]
        mountpoint = dev.get_mountpoint()
        if mountpoint:
            cmdline += [mountpoint]
        else:
            cmdline += [str(dev)]
        dummy, out, dummy = self.module.run_command(cmdline, check_rc=True, environ_update=self.LANG_ENV)
        block_size = block_count = None
        for line in out.splitlines():
            col = line.split('=')
            if col[0].strip() == 'data':
                if col[1].strip() == 'bsize':
                    block_size = int(col[2].split()[0])
                if col[2].split()[1] == 'blocks':
                    block_count = int(col[3].split(',')[0])
            if None not in (block_size, block_count):
                break
        else:
            raise ValueError(repr(out))
        return block_size * block_count