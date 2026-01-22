from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class StorageDomainModule(BaseModule):

    def _get_storage_type(self):
        for sd_type in ['nfs', 'iscsi', 'posixfs', 'glusterfs', 'fcp', 'localfs', 'managed_block_storage']:
            if self.param(sd_type) is not None:
                return sd_type

    def _get_storage(self):
        for sd_type in ['nfs', 'iscsi', 'posixfs', 'glusterfs', 'fcp', 'localfs', 'managed_block_storage']:
            if self.param(sd_type) is not None:
                return self.param(sd_type)

    def _get_storage_format(self):
        if self.param('storage_format') is not None:
            for sd_format in otypes.StorageFormat:
                if self.param('storage_format').lower() == str(sd_format):
                    return sd_format

    def _login(self, storage_type, storage):
        if storage_type == 'iscsi':
            hosts_service = self._connection.system_service().hosts_service()
            host_id = get_id_by_name(hosts_service, self.param('host'))
            if storage.get('target'):
                hosts_service.host_service(host_id).iscsi_login(iscsi=otypes.IscsiDetails(username=storage.get('username'), password=storage.get('password'), address=storage.get('address'), target=storage.get('target')))
            elif storage.get('target_lun_map'):
                for target in [m['target'] for m in storage.get('target_lun_map')]:
                    hosts_service.host_service(host_id).iscsi_login(iscsi=otypes.IscsiDetails(username=storage.get('username'), password=storage.get('password'), address=storage.get('address'), target=target))

    def __target_lun_map(self, storage):
        if storage.get('target'):
            lun_ids = storage.get('lun_id') if isinstance(storage.get('lun_id'), list) else [storage.get('lun_id')]
            return [(lun_id, storage.get('target')) for lun_id in lun_ids]
        elif storage.get('target_lun_map'):
            return [(target_map.get('lun_id'), target_map.get('target')) for target_map in storage.get('target_lun_map')]
        else:
            lun_ids = storage.get('lun_id') if isinstance(storage.get('lun_id'), list) else [storage.get('lun_id')]
            return [(lun_id, None) for lun_id in lun_ids]

    def build_entity(self):
        storage_type = self._get_storage_type()
        storage = self._get_storage()
        self._login(storage_type, storage)
        return otypes.StorageDomain(name=self.param('name'), description=self.param('description'), comment=self.param('comment'), wipe_after_delete=self.param('wipe_after_delete'), backup=self.param('backup'), critical_space_action_blocker=self.param('critical_space_action_blocker'), warning_low_space_indicator=self.param('warning_low_space'), import_=True if self.param('state') == 'imported' else None, id=self.param('id') if self.param('state') == 'imported' else None, type=otypes.StorageDomainType(storage_type if storage_type == 'managed_block_storage' else self.param('domain_function')), host=otypes.Host(name=self.param('host')), discard_after_delete=self.param('discard_after_delete'), storage=otypes.HostStorage(driver_options=[otypes.Property(name=do.get('name'), value=do.get('value')) for do in storage.get('driver_options')] if storage.get('driver_options') else None, driver_sensitive_options=[otypes.Property(name=dso.get('name'), value=dso.get('value')) for dso in storage.get('driver_sensitive_options')] if storage.get('driver_sensitive_options') else None, type=otypes.StorageType(storage_type), logical_units=[otypes.LogicalUnit(id=lun_id, address=storage.get('address'), port=int(storage.get('port', 3260)), target=target, username=storage.get('username'), password=storage.get('password')) for lun_id, target in self.__target_lun_map(storage)] if storage_type in ['iscsi', 'fcp'] else None, override_luns=storage.get('override_luns'), mount_options=storage.get('mount_options'), vfs_type='glusterfs' if storage_type in ['glusterfs'] else storage.get('vfs_type'), address=storage.get('address'), path=storage.get('path'), nfs_retrans=storage.get('retrans'), nfs_timeo=storage.get('timeout'), nfs_version=otypes.NfsVersion(storage.get('version')) if storage.get('version') else None) if storage_type is not None else None, storage_format=self._get_storage_format())

    def _find_attached_datacenter_name(self, sd_name):
        """
        Finds the name of the datacenter that a given
        storage domain is attached to.

        Args:
            sd_name (str): Storage Domain name

        Returns:
            str: Data Center name

        Raises:
            Exception: In case storage domain in not attached to
                an active Datacenter
        """
        dcs_service = self._connection.system_service().data_centers_service()
        dc = search_by_attributes(dcs_service, storage=sd_name)
        if dc is None:
            raise Exception("Can't bring storage to state `%s`, because it seems thatit is not attached to any datacenter" % self.param('state'))
        elif dc.status == dcstatus.UP:
            return dc.name
        else:
            raise Exception("Can't bring storage to state `%s`, because Datacenter %s is not UP" % (self.param('state'), dc.name))

    def _attached_sds_service(self, dc_name):
        dcs_service = self._connection.system_service().data_centers_service()
        dc = search_by_name(dcs_service, dc_name)
        if dc is None:
            dc = get_entity(dcs_service.service(dc_name))
            if dc is None:
                return None
        dc_service = dcs_service.data_center_service(dc.id)
        return dc_service.storage_domains_service()

    def _attached_sd_service(self, storage_domain):
        dc_name = self.param('data_center')
        if not dc_name:
            dc_name = self._find_attached_datacenter_name(storage_domain.name)
        attached_sds_service = self._attached_sds_service(dc_name)
        attached_sd_service = attached_sds_service.storage_domain_service(storage_domain.id)
        return attached_sd_service

    def _maintenance(self, storage_domain):
        attached_sd_service = self._attached_sd_service(storage_domain)
        attached_sd = get_entity(attached_sd_service)
        if attached_sd and attached_sd.status != sdstate.MAINTENANCE:
            if not self._module.check_mode:
                attached_sd_service.deactivate()
            self.changed = True
            wait(service=attached_sd_service, condition=lambda sd: sd.status == sdstate.MAINTENANCE, wait=self.param('wait'), timeout=self.param('timeout'))

    def _unattach(self, storage_domain):
        attached_sd_service = self._attached_sd_service(storage_domain)
        attached_sd = get_entity(attached_sd_service)
        if attached_sd and attached_sd.status == sdstate.MAINTENANCE:
            if not self._module.check_mode:
                attached_sd_service.remove()
            self.changed = True
            wait(service=attached_sd_service, condition=lambda sd: sd is None, wait=self.param('wait'), timeout=self.param('timeout'))

    def pre_remove(self, entity):
        if entity.status == sdstate.UNATTACHED or self.param('destroy'):
            return
        self._maintenance(entity)
        self._unattach(entity)

    def post_create_check(self, sd_id):
        storage_domain = self._service.service(sd_id).get()
        dc_name = self.param('data_center')
        if not dc_name:
            dc_name = self._find_attached_datacenter_name(storage_domain.name)
        self._service = self._attached_sds_service(dc_name)
        attached_sd_service = self._service.service(storage_domain.id)
        if get_entity(attached_sd_service) is None:
            self._service.add(otypes.StorageDomain(id=storage_domain.id))
            self.changed = True
            wait(service=attached_sd_service, condition=lambda sd: sd.status == sdstate.ACTIVE, wait=self.param('wait'), timeout=self.param('timeout'))

    def unattached_pre_action(self, storage_domain):
        dc_name = self.param('data_center')
        if not dc_name:
            dc_name = self._find_attached_datacenter_name(storage_domain.name)
        self._service = self._attached_sds_service(dc_name)
        self._maintenance(storage_domain)

    def update_check(self, entity):
        return equal(self.param('comment'), entity.comment) and equal(self.param('description'), entity.description) and equal(self.param('backup'), entity.backup) and equal(self.param('critical_space_action_blocker'), entity.critical_space_action_blocker) and equal(self.param('discard_after_delete'), entity.discard_after_delete) and equal(self.param('wipe_after_delete'), entity.wipe_after_delete) and equal(self.param('warning_low_space'), entity.warning_low_space_indicator)