from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def update_storage_class(module, current, patches):
    wanted = module.params
    if wanted['storage_class'] and wanted['storage_class'] != current.storage_class.name:
        patch = purefusion.VolumePatch(storage_class=purefusion.NullableString(wanted['storage_class']))
        patches.append(patch)