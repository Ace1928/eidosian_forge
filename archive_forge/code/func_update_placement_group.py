from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def update_placement_group(module, current, patches):
    wanted = module.params
    if wanted['placement_group'] and wanted['placement_group'] != current.placement_group.name:
        patch = purefusion.VolumePatch(placement_group=purefusion.NullableString(wanted['placement_group']))
        patches.append(patch)