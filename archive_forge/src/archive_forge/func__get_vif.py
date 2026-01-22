from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _get_vif(array, interface, subnet):
    """Return VLAN Interface or None"""
    vif_info = {}
    vif_name = interface['name'] + '.' + str(subnet['vlan'])
    try:
        interfaces = array.list_network_interfaces()
    except Exception:
        return None
    for ints in range(0, len(interfaces)):
        if interfaces[ints]['name'] == vif_name:
            vif_info = interfaces[ints]
            break
    return vif_info