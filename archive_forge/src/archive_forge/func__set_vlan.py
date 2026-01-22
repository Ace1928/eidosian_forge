from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _set_vlan(module):
    array = get_array(module)
    res = array.patch_hosts(names=[module.params['name']], host=flasharray.HostPatch(vlan=module.params['vlan']))
    if res.status_code != 200:
        module.warn('Failed to set host VLAN ID. Error: {0}'.format(res.errors[0].message))