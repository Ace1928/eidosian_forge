from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __execute_set_default_subvolume(self, operation):
    target = operation['target']
    target_id = operation['target_id']
    if target_id is None:
        target_subvolume = self.__filesystem.get_subvolume_by_name(target)
        if target_subvolume is None:
            self.__filesystem.refresh()
            target_subvolume = self.__filesystem.get_subvolume_by_name(target)
        if target_subvolume is None:
            raise BtrfsModuleException("Failed to find existing subvolume '%s'" % target)
        else:
            target_id = target_subvolume.id
    self.__btrfs_api.subvolume_set_default(self.__filesystem.get_any_mountpoint(), target_id)
    self.__completed_work.append(operation)