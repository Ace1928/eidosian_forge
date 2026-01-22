from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _update_vlan(module):
    changed = False
    array = get_array(module)
    host_vlan = getattr(list(array.get_hosts(names=[module.params['name']]).items)[0], 'vlan', None)
    if module.params['vlan'] != host_vlan:
        changed = True
        if not module.check_mode:
            res = array.patch_hosts(names=[module.params['name']], host=flasharray.HostPatch(vlan=module.params['vlan']))
            if res.status_code != 200:
                module.fail_json(msg='Failed to update host VLAN ID. Error: {0}'.format(res.errors[0].message))
    return changed