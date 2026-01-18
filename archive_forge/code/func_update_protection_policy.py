from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def update_protection_policy(module, current, patches):
    wanted = module.params
    current_policy = current.protection_policy.name if current.protection_policy else ''
    if wanted['protection_policy'] is not None and wanted['protection_policy'] != current_policy:
        patch = purefusion.VolumePatch(protection_policy=purefusion.NullableString(wanted['protection_policy']))
        patches.append(patch)