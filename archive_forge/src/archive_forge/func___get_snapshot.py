from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __get_snapshot(self):
    if self.param('snapshot_vm') is None:
        return None
    if self.param('snapshot_name') is None:
        return None
    vms_service = self._connection.system_service().vms_service()
    vm_id = get_id_by_name(vms_service, self.param('snapshot_vm'))
    vm_service = vms_service.vm_service(vm_id)
    snaps_service = vm_service.snapshots_service()
    snaps = snaps_service.list()
    snap = next((s for s in snaps if s.description == self.param('snapshot_name')), None)
    if not snap:
        raise ValueError('Snapshot with the name "{0}" was not found.'.format(self.param('snapshot_name')))
    return snap