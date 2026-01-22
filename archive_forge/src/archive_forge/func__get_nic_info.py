from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _get_nic_info(self):
    rv = {'network_info': []}
    vm_obj = self.get_vm()
    nic_info, nic_obj_lst = self._get_nics_from_vm(vm_obj)
    rv['network_info'] = nic_info
    return rv