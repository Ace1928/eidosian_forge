from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible.module_utils.basic import AnsibleModule
def update_destroyed(module, current, patches):
    wanted = module.params
    destroyed = wanted['state'] != 'present'
    if destroyed != current.destroyed:
        patch = purefusion.VolumePatch(destroyed=purefusion.NullableBoolean(destroyed))
        patches.append(patch)
        if destroyed and (not module.params['eradicate']):
            module.warn("Volume '{0}' is being soft deleted to prevent data loss, if you want to wipe it immediately to reclaim used space, add 'eradicate: true'".format(current.name))