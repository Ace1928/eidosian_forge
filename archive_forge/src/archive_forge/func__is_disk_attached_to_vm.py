from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _is_disk_attached_to_vm(self, vm_id, item):
    managed_by = item['managed_by']
    managed_by_extended = item['managed_by_extended']
    if managed_by is not None and vm_id == managed_by:
        return True
    if managed_by_extended is not None and vm_id in managed_by_extended:
        return True
    return False