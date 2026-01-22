from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _create_subinterfaces(module, array):
    subinterfaces_v1 = []
    subinterfaces_v2 = []
    all_children = True
    purity_vm = bool(len(array.get_controllers().items) == 1)
    if module.params['subinterfaces']:
        for inter in sorted(module.params['subinterfaces']):
            if array.get_network_interfaces(names=['ct0.' + inter]).status_code != 200:
                all_children = False
            if not all_children:
                module.fail_json(msg='Child subinterface {0} does not exist'.format(inter))
            subinterfaces_v2.append(FixedReferenceNoId(name='ct0.' + inter))
            subinterfaces_v1.append('ct0.' + inter)
            if not purity_vm:
                subinterfaces_v2.append(FixedReferenceNoId(name='ct1.' + inter))
                subinterfaces_v1.append('ct1.' + inter)
    return (subinterfaces_v1, subinterfaces_v2)