from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.snapshots import (
def update_pg(module, fusion, pg):
    """Update Placement Group"""
    pg_api_instance = purefusion.PlacementGroupsApi(fusion)
    patches = []
    update_display_name(module, fusion, patches, pg)
    update_array(module, fusion, patches, pg)
    if not module.check_mode:
        for patch in patches:
            op = pg_api_instance.update_placement_group(patch, tenant_name=module.params['tenant'], tenant_space_name=module.params['tenant_space'], placement_group_name=module.params['name'])
            await_operation(fusion, op)
    changed = len(patches) != 0
    return changed