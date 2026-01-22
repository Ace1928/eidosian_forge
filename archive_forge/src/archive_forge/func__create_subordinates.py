from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _create_subordinates(module, array):
    subordinates_v1 = []
    subordinates_v2 = []
    all_children = True
    if module.params['subordinates']:
        for inter in sorted(module.params['subordinates']):
            if array.get_network_interfaces(names=[inter]).status_code != 200:
                all_children = False
            if not all_children:
                module.fail_json(msg='Subordinate {0} does not exist. Ensure you have specified the controller.'.format(inter))
            subordinates_v2.append(FixedReferenceNoId(name=inter))
            subordinates_v1.append(inter)
    return (subordinates_v1, subordinates_v2)