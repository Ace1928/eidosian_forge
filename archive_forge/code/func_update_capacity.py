from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_capacity(module, array):
    """Expand CBS capacity"""
    steps = list(array.get_arrays_cloud_capacity_supported_steps().items)
    available = []
    for step in range(0, len(steps)):
        available.append(steps[step].supported_capacity)
    if module.params['capacity'] not in available:
        module.fail_json(msg='Selected capacity is not available. Run this module with `list` to get available capapcity points.')
    expanded = array.patch_arrays_cloud_capacity(capacity=flasharray.CloudCapacityStatus(requested_capacity=module.params['capacity']))
    if expanded.sttaus_code != 200:
        module.fail_json(msg='Expansion request failed. Error: {0}'.format(expanded.errors[0].message))