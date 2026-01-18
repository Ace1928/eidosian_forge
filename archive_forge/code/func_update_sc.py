from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def update_sc(module, fusion, s_class):
    """Update Storage Class settings"""
    changed = False
    sc_api_instance = purefusion.StorageClassesApi(fusion)
    if module.params['display_name'] and module.params['display_name'] != s_class.display_name:
        changed = True
        if not module.check_mode:
            sclass = purefusion.StorageClassPatch(display_name=purefusion.NullableString(module.params['display_name']))
            op = sc_api_instance.update_storage_class(sclass, storage_service_name=module.params['storage_service'], storage_class_name=module.params['name'])
            await_operation(fusion, op)
    module.exit_json(changed=changed, id=s_class.id)