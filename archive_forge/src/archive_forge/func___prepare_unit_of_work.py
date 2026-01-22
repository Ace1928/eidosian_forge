from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __prepare_unit_of_work(self):
    if self.__state == 'present':
        if self.__snapshot_source is None:
            self.__prepare_subvolume_present()
        else:
            self.__prepare_snapshot_present()
        if self.__default:
            self.__prepare_set_default()
    elif self.__state == 'absent':
        self.__prepare_subvolume_absent()