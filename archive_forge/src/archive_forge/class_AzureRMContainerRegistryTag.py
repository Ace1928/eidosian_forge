from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
class AzureRMContainerRegistryTag(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), registry=dict(type='str', required=True), repository_name=dict(type='str'), name=dict(type='str'), source_image=dict(type='dict', options=dict(registry_uri=dict(type='str'), repository=dict(type='str', required=True), name=dict(type='str', default='latest'), credentials=dict(type='dict', options=dict(username=dict(type='str'), password=dict(type='str', no_log=True))))), state=dict(type='str', default='present', choices=['present', 'absent']))
        required_if = [('state', 'present', ['source_image']), ('state', 'absent', ['repository_name'])]
        self.results = dict(changed=True)
        self.resource_group = None
        self.registry = None
        self.repository_name = None
        self.name = None
        self.source_image = None
        self.state = None
        self._client = None
        self._todo = Actions.NoAction
        super(AzureRMContainerRegistryTag, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=False, required_if=required_if)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        self._client = self.get_client()
        if self.state == 'present':
            repo_name = self.repository_name if self.repository_name else self.source_image['repository']
            tag_name = self.name if self.name else self.source_image['name']
            tag = self.get_tag(repo_name, tag_name)
            if not tag:
                self._todo = Actions.Import
        elif self.state == 'absent':
            if self.repository_name and self.name:
                tag = self.get_tag(self.repository_name, self.name)
                if tag:
                    self._todo = Actions.DeleteTag
            else:
                repository = self.get_repository(self.repository_name)
                if repository:
                    self._todo = Actions.DeleteRepo
        if self._todo == Actions.Import:
            self.log('importing image into registry')
            if not self.check_mode:
                self.import_tag(self.repository_name, self.name, self.resource_group, self.registry, self.source_image)
        elif self._todo == Actions.DeleteTag:
            self.log(f'deleting tag {self.repository_name}:{self.name}')
            if not self.check_mode:
                self.delete_tag(self.repository_name, self.name)
        elif self._todo == Actions.DeleteRepo:
            self.log(f'deleting repository {self.repository_name}')
            if not self.check_mode:
                self.delete_repository(self.repository_name)
        else:
            self.log('no action')
            self.results['changed'] = False
        return self.results

    def get_client(self):
        registry_endpoint = self.registry if self.registry.endswith('.azurecr.io') else self.registry + '.azurecr.io'
        return ContainerRegistryClient(endpoint=registry_endpoint, credential=self.azure_auth.azure_credential_track2, audience='https://management.azure.com')

    def get_repository(self, repository_name):
        response = None
        try:
            response = self._client.get_repository_properties(repository=repository_name)
            self.log(f'Response : {response}')
        except Exception as e:
            self.log(f'Could not get ACR repository for {repository_name} - {str(e)}')
        if response is not None:
            return response.name
        return None

    def get_tag(self, repository_name, tag_name):
        response = None
        try:
            self.log(f'Getting tag for {repository_name}:{tag_name}')
            response = self._client.get_tag_properties(repository=repository_name, tag=tag_name)
            self.log(f'Response : {response}')
        except Exception as e:
            self.log(f'Could not get ACR tag for {repository_name}:{tag_name} - {str(e)}')
        return response

    def import_tag(self, repository, tag, resource_group, registry, source_image):
        source_tag = get_tag(source_image['repository'], source_image['name'])
        dest_repo_name = repository if repository else source_image['repository']
        dest_tag_name = tag if tag else source_image['name']
        dest_tag = get_tag(dest_repo_name, dest_tag_name)
        creds = None if not source_image['credentials'] else ImportSourceCredentials(username=source_image['credentials']['username'], password=source_image['credentials']['password'])
        params = ImportImageParameters(target_tags=[dest_tag], source=ImportSource(registry_uri=source_image['registry_uri'], source_image=source_tag, credentials=creds))
        try:
            if not resource_group:
                resource_group = self.get_registry_resource_group(registry)
            self.log(f'Importing {source_tag} as {dest_tag} to {registry} in {resource_group}')
            poller = self.containerregistry_client.registries.begin_import_image(resource_group_name=resource_group, registry_name=registry, parameters=params)
            self.get_poller_result(poller)
        except Exception as e:
            self.fail(f'Could not import {source_tag} as {dest_tag} to {registry} in {resource_group} - {str(e)}')

    def get_registry_resource_group(self, registry_name):
        response = None
        try:
            response = self.containerregistry_client.registries.list()
        except Exception as e:
            self.fail(f'Could not load resource group for registry {registry_name} - {str(e)}')
        if response is not None:
            for item in response:
                item_dict = item.as_dict()
                if item_dict['name'] == registry_name:
                    return azure_id_to_dict(item_dict['id']).get('resourceGroups')
        return None

    def delete_repository(self, repository_name):
        try:
            self._client.delete_repository(repository=repository_name)
        except Exception as e:
            self.fail(f'Could not delete repository {repository_name} - {str(e)}')

    def delete_tag(self, repository_name, tag_name):
        try:
            self._client.delete_tag(repository=repository_name, tag=tag_name)
        except Exception as e:
            self.fail(f'Could not delete tag {repository_name}:{tag_name} - {str(e)}')