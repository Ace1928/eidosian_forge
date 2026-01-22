from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __find_matching_filesytem(self):
    criteria = {'uuid': self.__filesystem_uuid, 'label': self.__filesystem_label, 'device': self.__filesystem_device}
    return self.__provider.get_matching_filesystem(criteria)