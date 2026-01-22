from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __stage_required_mount(self, subvolume):
    if subvolume.get_mounted_path() is None:
        if self.__automount:
            self.__required_mounts.append(subvolume)
        else:
            raise BtrfsModuleException("The requested changes will require the subvolume '%s' to be mounted, but automount=False" % subvolume.path)