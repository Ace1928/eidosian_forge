from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _attach_cd(self, entity):
    cd_iso_id = self.param('cd_iso')
    if cd_iso_id is not None:
        if cd_iso_id:
            cd_iso_id = self.__get_cd_id()
        vm_service = self._service.service(entity.id)
        current = vm_service.get().status == otypes.VmStatus.UP and self.param('state') == 'running'
        cdroms_service = vm_service.cdroms_service()
        cdrom_device = cdroms_service.list()[0]
        cdrom_service = cdroms_service.cdrom_service(cdrom_device.id)
        cdrom = cdrom_service.get(current=current)
        if getattr(cdrom.file, 'id', '') != cd_iso_id:
            if not self._module.check_mode:
                cdrom_service.update(cdrom=otypes.Cdrom(file=otypes.File(id=cd_iso_id)), current=current)
            self.changed = True
    return entity