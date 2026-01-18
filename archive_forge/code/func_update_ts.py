from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils import getters
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def update_ts(module, fusion, ts):
    """Update Tenant Space"""
    ts_api_instance = purefusion.TenantSpacesApi(fusion)
    patches = []
    if module.params['display_name'] and module.params['display_name'] != ts.display_name:
        patch = purefusion.TenantSpacePatch(display_name=purefusion.NullableString(module.params['display_name']))
        patches.append(patch)
    if not module.check_mode:
        for patch in patches:
            op = ts_api_instance.update_tenant_space(patch, tenant_name=module.params['tenant'], tenant_space_name=module.params['name'])
            await_operation(fusion, op)
    changed = len(patches) != 0
    module.exit_json(changed=changed, id=ts.id)