from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def update_vms(self, affinity_group):
    """
        This method iterate via the affinity VM assignments and datech the VMs
        which should not be attached to affinity and attach VMs which should be
        attached to affinity.
        """
    assigned_vms = self.assigned_vms(affinity_group)
    to_remove = list((vm for vm in assigned_vms if vm not in self._vm_ids))
    to_add = []
    if self._vm_ids:
        to_add = list((vm for vm in self._vm_ids if vm not in assigned_vms))
    ag_service = self._service.group_service(affinity_group.id)
    for vm in to_remove:
        ag_service.vms_service().vm_service(vm).remove()
    for vm in to_add:
        try:
            ag_service.vms_service().add(otypes.Vm(id=vm))
        except ValueError as ex:
            if 'complete' not in str(ex):
                raise ex