from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __stage_create_snapshot(self, source_subvolume, target_subvolume_path):
    """Add creation of a snapshot from source to target to the unit of work"""
    self.__unit_of_work.append({'action': self.__CREATE_SNAPSHOT_OPERATION, 'source': source_subvolume.path, 'source_id': source_subvolume.id, 'target': target_subvolume_path})