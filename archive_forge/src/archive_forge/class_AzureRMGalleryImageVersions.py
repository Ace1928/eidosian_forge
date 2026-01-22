from __future__ import absolute_import, division, print_function
import time
import json
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
class AzureRMGalleryImageVersions(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', updatable=False, disposition='resourceGroupName', required=True), gallery_name=dict(type='str', updatable=False, disposition='galleryName', required=True), gallery_image_name=dict(type='str', updatable=False, disposition='galleryImageName', required=True), name=dict(type='str', updatable=False, disposition='galleryImageVersionName', required=True), tags=dict(type='dict', updatable=False, disposition='tags', comparison='tags'), location=dict(type='str', updatable=False, disposition='/', comparison='location'), storage_profile=dict(type='dict', updatable=False, disposition='/properties/storageProfile', comparison='ignore', options=dict(source_image=dict(type='raw', disposition='source/id', purgeIfNone=True, pattern=['/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/images/{name}', '/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/galleries/{gallery_name}/images/{gallery_image_name}/versions/{version}']), os_disk=dict(type='dict', disposition='osDiskImage', purgeIfNone=True, comparison='ignore', options=dict(source=dict(type='raw', disposition='source/id', pattern='/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/snapshots/{name}'), host_caching=dict(type='str', disposition='hostCaching', default='None', choices=['ReadOnly', 'ReadWrite', 'None']))), data_disks=dict(type='list', elements='raw', disposition='dataDiskImages', purgeIfNone=True, options=dict(lun=dict(type='int'), source=dict(type='raw', disposition='source/id', pattern='/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/snapshots/{name}'), host_caching=dict(type='str', disposition='hostCaching', default='None', choices=['ReadOnly', 'ReadWrite', 'None']))))), publishing_profile=dict(type='dict', disposition='/properties/publishingProfile', options=dict(target_regions=dict(type='list', elements='raw', disposition='targetRegions', options=dict(name=dict(type='str', required=True, comparison='location'), regional_replica_count=dict(type='int', disposition='regionalReplicaCount'), storage_account_type=dict(type='str', disposition='storageAccountType'))), managed_image=dict(type='raw', pattern='/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/images/{name}', comparison='ignore'), snapshot=dict(type='raw', pattern='/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/snapshots/{name}', comparison='ignore'), replica_count=dict(type='int', disposition='replicaCount'), exclude_from_latest=dict(type='bool', disposition='excludeFromLatest'), end_of_life_date=dict(type='str', disposition='endOfLifeDate'), storage_account_type=dict(type='str', disposition='storageAccountType', choices=['Standard_LRS', 'Standard_ZRS']))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.gallery_name = None
        self.gallery_image_name = None
        self.name = None
        self.gallery_image_version = None
        self.tags = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200, 201, 202]
        self.to_do = Actions.NoAction
        self.body = {}
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2019-07-01'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureRMGalleryImageVersions, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
                if key == 'tags':
                    self.body[key] = kwargs[key]
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        snapshot = self.body.get('properties', {}).get('publishingProfile', {}).pop('snapshot', None)
        if snapshot is not None:
            self.body['properties'].setdefault('storageProfile', {}).setdefault('osDiskImage', {}).setdefault('source', {})['id'] = snapshot
        managed_image = self.body.get('properties', {}).get('publishingProfile', {}).pop('managed_image', None)
        if managed_image:
            self.body['properties'].setdefault('storageProfile', {}).setdefault('source', {})['id'] = managed_image
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.body:
            self.body['location'] = resource_group.location
        self.url = '/subscriptions' + '/{{ subscription_id }}' + '/resourceGroups' + '/{{ resource_group }}' + '/providers' + '/Microsoft.Compute' + '/galleries' + '/{{ gallery_name }}' + '/images' + '/{{ image_name }}' + '/versions' + '/{{ version_name }}'
        self.url = self.url.replace('{{ subscription_id }}', self.subscription_id)
        self.url = self.url.replace('{{ resource_group }}', self.resource_group)
        self.url = self.url.replace('{{ gallery_name }}', self.gallery_name)
        self.url = self.url.replace('{{ image_name }}', self.gallery_image_name)
        self.url = self.url.replace('{{ version_name }}', self.name)
        old_response = self.get_resource()
        if not old_response:
            self.log("GalleryImageVersion instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('GalleryImageVersion instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            else:
                update_tags, newtags = self.update_tags(old_response.get('tags', dict()))
                if update_tags:
                    self.tags = newtags
                    self.body['tags'] = self.tags
                    self.to_do = Actions.Update
                modifiers = {}
                self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
                self.results['modifiers'] = modifiers
                self.results['compare'] = []
                if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the GalleryImageVersion instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_resource()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('GalleryImageVersion instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_resource()
        else:
            self.log('GalleryImageVersion instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
        return self.results

    def create_update_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as exc:
            self.log('Error attempting to create the GalleryImageVersion instance.')
            self.fail('Error creating the GalleryImageVersion instance: {0}'.format(str(exc)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        while response['properties']['provisioningState'] == 'Creating':
            time.sleep(60)
            response = self.get_resource()
        return response

    def delete_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
        except Exception as e:
            self.log('Error attempting to delete the GalleryImageVersion instance.')
            self.fail('Error deleting the GalleryImageVersion instance: {0}'.format(str(e)))
        return True

    def get_resource(self):
        found = False
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            response = json.loads(response.body())
            found = True
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Did not find the AzureFirewall instance.')
        if found is True:
            return response
        return False