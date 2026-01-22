from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDataFactoryInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), if_none_match=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.if_none_match = None
        self.tags = None
        super(AzureRMDataFactoryInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        result = []
        if self.name and self.resource_group:
            result = self.get_item()
        elif self.resource_group:
            result = self.list_by_resourcegroup()
        else:
            result = self.list_all()
        self.results['datafactory'] = self.format(result)
        return self.results

    def format(self, raw):
        results = []
        for item in raw:
            if self.has_tags(item.tags, self.tags):
                results.append(self.pip_to_dict(item))
        return results

    def pip_to_dict(self, pip):
        result = dict(id=pip.id, name=pip.name, type=pip.type, location=pip.location, tags=pip.tags, e_tag=pip.e_tag, provisioning_state=pip.provisioning_state, create_time=pip.create_time, repo_configuration=dict(), identity=dict(), public_network_access=pip.public_network_access)
        if pip.identity:
            result['identity']['principal_id'] = pip.identity.principal_id
            result['identity']['tenant_id'] = pip.identity.tenant_id
        if pip.repo_configuration:
            result['repo_configuration']['type'] = pip.repo_configuration.type
            result['repo_configuration']['account_name'] = pip.repo_configuration.account_name
            result['repo_configuration']['repository_name'] = pip.repo_configuration.repository_name
            result['repo_configuration']['collaboration_branch'] = pip.repo_configuration.collaboration_branch
            result['repo_configuration']['root_folder'] = pip.repo_configuration.root_folder
            if pip.repo_configuration.type == 'FactoryVSTSConfiguration':
                result['repo_configuration']['project_name'] = pip.repo_configuration.project_name
        return result

    def get_item(self):
        response = None
        self.log('Get properties for {0}'.format(self.name))
        try:
            response = self.datafactory_client.factories.get(self.resource_group, self.name, self.if_none_match)
        except ResourceNotFoundError:
            pass
        return [response] if response else []

    def list_by_resourcegroup(self):
        self.log('Get GitHub Access Token Response')
        try:
            response = self.datafactory_client.factories.list_by_resource_group(self.resource_group)
        except Exception:
            pass
        return response if response else []

    def list_all(self):
        self.log('Get GitHub Access Token Response')
        try:
            response = self.datafactory_client.factories.list()
        except Exception:
            pass
        return response if response else []