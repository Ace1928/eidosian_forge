from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __format_delete_subvolume_result(self, operation):
    target = operation['target']
    target_id = operation['target_id']
    return "Deleted subvolume '%s' (%s)" % (target, target_id)