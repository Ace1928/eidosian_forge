from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def update_change_flag(standby_changed, performance_changed, renamed_changed, interface_changed, mode_changed, remove_changed, owner_changed):
    """ Update the changed flag based on the operation performed in the task"""
    if standby_changed or performance_changed or renamed_changed or interface_changed or mode_changed or remove_changed or owner_changed:
        return True
    return False