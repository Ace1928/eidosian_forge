from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __prepare_subvolume_absent(self):
    subvolume = self.__filesystem.get_subvolume_by_name(self.__name)
    if subvolume is not None:
        self.__prepare_delete_subvolume_tree(subvolume)