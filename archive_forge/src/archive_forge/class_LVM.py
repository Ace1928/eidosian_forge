from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class LVM(Filesystem):
    MKFS = 'pvcreate'
    MKFS_FORCE_FLAGS = ['-f']
    MKFS_SET_UUID_OPTIONS = ['-u', '--uuid']
    MKFS_SET_UUID_EXTRA_OPTIONS = ['--norestorefile']
    INFO = 'pvs'
    GROW = 'pvresize'
    CHANGE_UUID = 'pvchange'
    CHANGE_UUID_OPTION = '-u'
    CHANGE_UUID_OPTION_HAS_ARG = False

    def get_fs_size(self, dev):
        """Get and return PV size, in bytes."""
        cmd = self.module.get_bin_path(self.INFO, required=True)
        dummy, size, dummy = self.module.run_command([cmd, '--noheadings', '-o', 'pv_size', '--units', 'b', '--nosuffix', str(dev)], check_rc=True)
        pv_size = int(size)
        return pv_size