from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __is_existing_directory_like(self, path):
    return os.path.exists(path) and (os.path.isdir(path) or os.stat(path).st_ino == self.__BTRFS_SUBVOLUME_INODE_NUMBER)