from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __get_numa_serialized(self, numa):
    return sorted([(x.index, [y.index for y in x.cpu.cores] if x.cpu else [], x.memory, [y.index for y in x.numa_node_pins] if x.numa_node_pins else []) for x in numa], key=lambda x: x[0])