from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __check_required_mounts(self):
    filtered = self.__filter_child_subvolumes(self.__required_mounts)
    if len(filtered) > 0:
        for subvolume in filtered:
            self.__mount_subvolume_id_to_tempdir(self.__filesystem, subvolume.id)
        self.__filesystem.refresh_mountpoints()