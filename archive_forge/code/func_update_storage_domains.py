from __future__ import (absolute_import, division, print_function)
import json
import os
import subprocess
import time
import traceback
import inspect
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def update_storage_domains(self, disk_id):
    changed = False
    disk_service = self._service.service(disk_id)
    disk = disk_service.get()
    sds_service = self._connection.system_service().storage_domains_service()
    if disk.storage_type != otypes.DiskStorageType.IMAGE:
        return changed
    if disk.content_type in [otypes.DiskContentType(x) for x in ['hosted_engine', 'hosted_engine_sanlock', 'hosted_engine_metadata', 'hosted_engine_configuration']]:
        return changed
    if self._module.params['storage_domain']:
        new_disk_storage_id = get_id_by_name(sds_service, self._module.params['storage_domain'])
        if new_disk_storage_id in [sd.id for sd in disk.storage_domains]:
            return changed
        changed = self.action(action='move', entity=disk, action_condition=lambda d: new_disk_storage_id != d.storage_domains[0].id, wait_condition=lambda d: d.status == otypes.DiskStatus.OK, storage_domain=otypes.StorageDomain(id=new_disk_storage_id), post_action=lambda _: time.sleep(self._module.params['poll_interval']))['changed']
    if self._module.params['storage_domains']:
        for sd in self._module.params['storage_domains']:
            new_disk_storage = search_by_name(sds_service, sd)
            changed = changed or self.action(action='copy', entity=disk, action_condition=lambda d: new_disk_storage.id not in [sd.id for sd in d.storage_domains], wait_condition=lambda d: d.status == otypes.DiskStatus.OK, storage_domain=otypes.StorageDomain(id=new_disk_storage.id))['changed']
    return changed