from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __prepare_delete_subvolume_tree(self, subvolume):
    if subvolume.is_filesystem_root():
        raise BtrfsModuleException("Can not delete the filesystem's root subvolume")
    if not self.__recursive and len(subvolume.get_child_subvolumes()) > 0:
        raise BtrfsModuleException('Subvolume targeted for deletion %s has children and recursive=False.Either explicitly delete the child subvolumes first or pass parameter recursive=True.' % subvolume.path)
    self.__stage_required_mount(subvolume.get_parent_subvolume())
    queue = self.__prepare_recursive_delete_order(subvolume) if self.__recursive else [subvolume]
    for s in queue:
        if s.is_mounted():
            raise BtrfsModuleException('Can not delete mounted subvolume=%s' % s.path)
        if s.is_filesystem_default():
            self.__stage_set_default_subvolume(self.__BTRFS_ROOT_SUBVOLUME, self.__BTRFS_ROOT_SUBVOLUME_ID)
        self.__stage_delete_subvolume(s)