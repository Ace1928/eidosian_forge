from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def update_connection(module, blade, target_blade):
    """Update array connection - only encryption currently"""
    changed = False
    if target_blade.management_address is None:
        module.fail_json(msg='Update can only happen from the array that formed the connection')
    if module.params['encrypted'] != target_blade.encrypted:
        if module.params['encrypted'] and blade.file_system_replica_links.list_file_system_replica_links().pagination_info.total_item_count != 0:
            module.fail_json(msg='Cannot turn array connection encryption on if file system replica links exist')
        new_attr = ArrayConnectionv1(encrypted=module.params['encrypted'])
        changed = True
        if not module.check_mode:
            try:
                blade.array_connections.update_array_connections(remote_names=[target_blade.remote.name], array_connection=new_attr)
            except Exception:
                module.fail_json(msg='Failed to change encryption setting for array connection.')
    module.exit_json(changed=changed)