from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.snapshots import (
def update_display_name(module, fusion, patches, pg):
    if not module.params['display_name']:
        return
    if module.params['display_name'] == pg.display_name:
        return
    patch = purefusion.PlacementGroupPatch(display_name=purefusion.NullableString(module.params['display_name']))
    patches.append(patch)