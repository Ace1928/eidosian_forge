from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
class AzureRMImage(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), source=dict(type='raw'), data_disk_sources=dict(type='list', elements='str', default=[]), os_type=dict(type='str', choices=['Windows', 'Linux']), hyper_v_generation=dict(type='str', choices=['V1', 'V2']))
        self.results = dict(changed=False, id=None)
        required_if = [('state', 'present', ['source'])]
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.source = None
        self.data_disk_sources = None
        self.os_type = None
        self.hyper_v_generation = None
        super(AzureRMImage, self).__init__(self.module_arg_spec, supports_check_mode=True, required_if=required_if)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        results = None
        changed = False
        image = None
        if not self.location:
            resource_group = self.get_resource_group(self.resource_group)
            self.location = resource_group.location
        self.log('Fetching image {0}'.format(self.name))
        image = self.get_image()
        if image:
            self.check_provisioning_state(image, self.state)
            results = image.id
            update_tags, tags = self.update_tags(image.tags)
            if update_tags:
                changed = True
                self.tags = tags
            if self.hyper_v_generation and self.hyper_v_generation != image.hyper_v_generation:
                self.log('Compare configure Check whether hyper_v_generation needs to be updated')
                self.fail('The hyper_v_generation parameter cannot be updated to {0}'.format(self.hyper_v_generation))
            else:
                self.hyper_v_generation = image.hyper_v_generation
            if self.state == 'absent':
                changed = True
        elif self.state == 'present':
            changed = True
        self.results['changed'] = changed
        self.results['id'] = results
        if changed:
            if self.state == 'present':
                image_instance = None
                vm = self.get_source_vm()
                if vm:
                    if self.data_disk_sources:
                        self.fail('data_disk_sources is not allowed when capturing image from vm')
                    image_instance = self.image_models.Image(location=self.location, source_virtual_machine=self.image_models.SubResource(id=vm.id), hyper_v_generation=self.hyper_v_generation, tags=self.tags)
                else:
                    if not self.os_type:
                        self.fail('os_type is required to create the image')
                    os_disk = self.create_os_disk()
                    data_disks = self.create_data_disks()
                    storage_profile = self.image_models.ImageStorageProfile(os_disk=os_disk, data_disks=data_disks)
                    image_instance = self.image_models.Image(location=self.location, storage_profile=storage_profile, hyper_v_generation=self.hyper_v_generation, tags=self.tags)
                if not self.check_mode and image_instance:
                    new_image = self.create_image(image_instance)
                    self.results['id'] = new_image.id
            elif self.state == 'absent':
                if not self.check_mode:
                    self.delete_image()
                    self.results['id'] = None
        return self.results

    def resolve_storage_source(self, source):
        blob_uri = None
        disk = None
        snapshot = None
        if isinstance(source, str) and source.lower().endswith('.vhd'):
            blob_uri = source
            return (blob_uri, disk, snapshot)
        tokenize = dict()
        if isinstance(source, dict):
            tokenize = source
        elif isinstance(source, str):
            tokenize = parse_resource_id(source)
        else:
            self.fail('source parameter should be in type string or dictionary')
        if tokenize.get('type') == 'disks':
            disk = format_resource_id(tokenize['name'], tokenize.get('subscription_id') or self.subscription_id, 'Microsoft.Compute', 'disks', tokenize.get('resource_group') or self.resource_group)
            return (blob_uri, disk, snapshot)
        if tokenize.get('type') == 'snapshots':
            snapshot = format_resource_id(tokenize['name'], tokenize.get('subscription_id') or self.subscription_id, 'Microsoft.Compute', 'snapshots', tokenize.get('resource_group') or self.resource_group)
            return (blob_uri, disk, snapshot)
        if 'type' in tokenize:
            return (blob_uri, disk, snapshot)
        snapshot_instance = self.get_snapshot(tokenize.get('resource_group') or self.resource_group, tokenize['name'])
        if snapshot_instance:
            snapshot = snapshot_instance.id
            return (blob_uri, disk, snapshot)
        disk_instance = self.get_disk(tokenize.get('resource_group') or self.resource_group, tokenize['name'])
        if disk_instance:
            disk = disk_instance.id
        return (blob_uri, disk, snapshot)

    def create_os_disk(self):
        blob_uri, disk, snapshot = self.resolve_storage_source(self.source)
        snapshot_resource = self.image_models.SubResource(id=snapshot) if snapshot else None
        managed_disk = self.image_models.SubResource(id=disk) if disk else None
        return self.image_models.ImageOSDisk(os_type=self.os_type, os_state=self.image_models.OperatingSystemStateTypes.generalized, snapshot=snapshot_resource, managed_disk=managed_disk, blob_uri=blob_uri)

    def create_data_disk(self, lun, source):
        blob_uri, disk, snapshot = self.resolve_storage_source(source)
        if blob_uri or disk or snapshot:
            snapshot_resource = self.image_models.SubResource(id=snapshot) if snapshot else None
            managed_disk = self.image_models.SubResource(id=disk) if disk else None
            return self.image_models.ImageDataDisk(lun=lun, blob_uri=blob_uri, snapshot=snapshot_resource, managed_disk=managed_disk)

    def create_data_disks(self):
        return list(filter(None, [self.create_data_disk(lun, source) for lun, source in enumerate(self.data_disk_sources)]))

    def get_source_vm(self):
        resource = dict()
        if isinstance(self.source, dict):
            if self.source.get('type') != 'virtual_machines':
                return None
            resource = dict(type='virtualMachines', name=self.source['name'], resource_group=self.source.get('resource_group') or self.resource_group)
        elif isinstance(self.source, str):
            vm_resource_id = format_resource_id(self.source, self.subscription_id, 'Microsoft.Compute', 'virtualMachines', self.resource_group)
            resource = parse_resource_id(vm_resource_id)
        else:
            self.fail('Unsupported type of source parameter, please give string or dictionary')
        return self.get_vm(resource['resource_group'], resource['name']) if resource['type'] == 'virtualMachines' else None

    def get_snapshot(self, resource_group, snapshot_name):
        return self._get_resource(self.image_client.snapshots.get, resource_group, snapshot_name)

    def get_disk(self, resource_group, disk_name):
        return self._get_resource(self.image_client.disks.get, resource_group, disk_name)

    def get_vm(self, resource_group, vm_name):
        return self._get_resource(self.image_client.virtual_machines.get, resource_group, vm_name, 'instanceview')

    def get_image(self):
        return self._get_resource(self.image_client.images.get, self.resource_group, self.name)

    def _get_resource(self, get_method, resource_group, name, expand=None):
        try:
            if expand:
                return get_method(resource_group, name, expand=expand)
            else:
                return get_method(resource_group, name)
        except ResourceNotFoundError as cloud_err:
            if cloud_err.status_code == 404:
                self.log('{0}'.format(str(cloud_err)))
                return None
            self.fail('Error: failed to get resource {0} - {1}'.format(name, str(cloud_err)))

    def create_image(self, image):
        try:
            poller = self.image_client.images.begin_create_or_update(self.resource_group, self.name, image)
            new_image = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error creating image {0} - {1}'.format(self.name, str(exc)))
        self.check_provisioning_state(new_image)
        return new_image

    def delete_image(self):
        self.log('Deleting image {0}'.format(self.name))
        try:
            poller = self.image_client.images.begin_delete(self.resource_group, self.name)
            result = self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting image {0} - {1}'.format(self.name, str(exc)))
        return result