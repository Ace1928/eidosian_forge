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
class DisksModule(BaseModule):

    def build_entity(self):
        hosts_service = self._connection.system_service().hosts_service()
        logical_unit = self._module.params.get('logical_unit')
        size = convert_to_bytes(self._module.params.get('size'))
        if not size and self._module.params.get('upload_image_path'):
            out = subprocess.check_output(['qemu-img', 'info', '--output', 'json', self._module.params.get('upload_image_path')])
            image_info = json.loads(out)
            size = image_info['virtual-size']
        disk = otypes.Disk(id=self._module.params.get('id'), name=self._module.params.get('name'), description=self._module.params.get('description'), format=otypes.DiskFormat(self._module.params.get('format')) if self._module.params.get('format') else None, content_type=otypes.DiskContentType(self._module.params.get('content_type')) if self._module.params.get('content_type') else None, sparse=self._module.params.get('sparse') if self._module.params.get('sparse') is not None else self._module.params.get('format') != 'raw', openstack_volume_type=otypes.OpenStackVolumeType(name=self.param('openstack_volume_type')) if self.param('openstack_volume_type') else None, provisioned_size=size, storage_domains=[otypes.StorageDomain(name=self._module.params.get('storage_domain'))], disk_profile=otypes.DiskProfile(id=get_id_by_name(self._connection.system_service().disk_profiles_service(), self._module.params.get('profile'))) if self._module.params.get('profile') else None, quota=otypes.Quota(id=self._module.params.get('quota_id')) if self.param('quota_id') else None, shareable=self._module.params.get('shareable'), sgio=otypes.ScsiGenericIO(self.param('scsi_passthrough')) if self.param('scsi_passthrough') else None, propagate_errors=self.param('propagate_errors'), backup=otypes.DiskBackup(self.param('backup')) if self.param('backup') else None, wipe_after_delete=self.param('wipe_after_delete'), lun_storage=otypes.HostStorage(host=otypes.Host(id=get_id_by_name(hosts_service, self._module.params.get('host'))) if self.param('host') else None, type=otypes.StorageType(logical_unit.get('storage_type', 'iscsi')), logical_units=[otypes.LogicalUnit(address=logical_unit.get('address'), port=logical_unit.get('port', 3260), target=logical_unit.get('target'), id=logical_unit.get('id'), username=logical_unit.get('username'), password=logical_unit.get('password'))]) if logical_unit else None)
        if hasattr(disk, 'initial_size') and self._module.params['upload_image_path']:
            out = subprocess.check_output(['qemu-img', 'measure', '-O', 'qcow2' if self._module.params.get('format') == 'cow' else 'raw', '--output', 'json', self._module.params['upload_image_path']])
            measure = json.loads(out)
            disk.initial_size = measure['required']
        return disk

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

    def update_check(self, entity):
        return equal(self._module.params.get('name'), entity.name) and equal(self._module.params.get('description'), entity.description) and equal(self.param('quota_id'), getattr(entity.quota, 'id', None)) and equal(convert_to_bytes(self._module.params.get('size')), entity.provisioned_size) and equal(self._module.params.get('shareable'), entity.shareable) and equal(self.param('propagate_errors'), entity.propagate_errors) and equal(otypes.ScsiGenericIO(self.param('scsi_passthrough')) if self.param('scsi_passthrough') else None, entity.sgio) and equal(self.param('wipe_after_delete'), entity.wipe_after_delete) and equal(self.param('profile'), getattr(follow_link(self._connection, entity.disk_profile), 'name', None))