from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.btrfs import BtrfsFilesystemsProvider, BtrfsCommands, BtrfsModuleException
from ansible_collections.community.general.plugins.module_utils.btrfs import normalize_subvolume_path
from ansible.module_utils.basic import AnsibleModule
import os
import tempfile
def __prepare_recursive_delete_order(self, subvolume):
    """Return the subvolume and all descendents as a list, ordered so that descendents always occur before their ancestors"""
    pending = [subvolume]
    ordered = []
    while len(pending) > 0:
        next = pending.pop()
        ordered.append(next)
        pending.extend(next.get_child_subvolumes())
    ordered.reverse()
    return ordered