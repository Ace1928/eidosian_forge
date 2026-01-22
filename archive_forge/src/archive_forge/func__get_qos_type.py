from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_qos_type(self, type):
    if type == 'storage':
        return otypes.QosType.STORAGE
    elif type == 'network':
        return otypes.QosType.NETWORK
    elif type == 'hostnetwork':
        return otypes.QosType.HOSTNETWORK
    elif type == 'cpu':
        return otypes.QosType.CPU
    return None