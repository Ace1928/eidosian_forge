from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _get_interface(module, array):
    """Return Network Interface or None"""
    interface = {}
    if module.params['name'][0] == 'v':
        try:
            interface = array.get_network_interface(module.params['name'])
        except Exception:
            return None
    else:
        try:
            interfaces = array.list_network_interfaces()
        except Exception:
            return None
        for ints in range(0, len(interfaces)):
            if interfaces[ints]['name'] == module.params['name']:
                interface = interfaces[ints]
                break
    return interface