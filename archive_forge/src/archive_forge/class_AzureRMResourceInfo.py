from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
class AzureRMResourceInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(url=dict(type='str'), provider=dict(type='str'), resource_group=dict(type='str'), resource_type=dict(type='str'), resource_name=dict(type='str'), subresource=dict(type='list', elements='dict', default=[], options=dict(namespace=dict(type='str'), type=dict(type='str'), name=dict(type='str'))), method=dict(type='str', default='GET', choices=['GET', 'PUT', 'POST', 'HEAD', 'PATCH', 'DELETE', 'MERGE']), api_version=dict(type='str'))
        self.results = dict(response=[])
        self.mgmt_client = None
        self.url = None
        self.api_version = None
        self.provider = None
        self.resource_group = None
        self.resource_type = None
        self.resource_name = None
        self.subresource = []
        super(AzureRMResourceInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_resource_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_resource_facts' module has been renamed to 'azure_rm_resource_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.url is None:
            orphan = None
            rargs = dict()
            rargs['subscription'] = self.subscription_id
            rargs['resource_group'] = self.resource_group
            if not (self.provider is None or self.provider.lower().startswith('.microsoft')):
                rargs['namespace'] = 'Microsoft.' + self.provider
            else:
                rargs['namespace'] = self.provider
            if self.resource_type is not None and self.resource_name is not None:
                rargs['type'] = self.resource_type
                rargs['name'] = self.resource_name
                for i in range(len(self.subresource)):
                    resource_ns = self.subresource[i].get('namespace', None)
                    resource_type = self.subresource[i].get('type', None)
                    resource_name = self.subresource[i].get('name', None)
                    if resource_type is not None and resource_name is not None:
                        rargs['child_namespace_' + str(i + 1)] = resource_ns
                        rargs['child_type_' + str(i + 1)] = resource_type
                        rargs['child_name_' + str(i + 1)] = resource_name
                    else:
                        orphan = resource_type
            else:
                orphan = self.resource_type
            self.url = resource_id(**rargs)
            if orphan is not None:
                self.url += '/' + orphan
        if not self.api_version:
            try:
                if '/providers/' in self.url:
                    provider = self.url.split('/providers/')[1].split('/')[0]
                    resourceType = self.url.split(provider + '/')[1].split('/')[0]
                    url = '/subscriptions/' + self.subscription_id + '/providers/' + provider
                    api_versions = json.loads(self.mgmt_client.query(url, self.method, {'api-version': '2015-01-01'}, None, None, [200], 0, 0).body())
                    for rt in api_versions['resourceTypes']:
                        if rt['resourceType'].lower() == resourceType.lower():
                            self.api_version = rt['apiVersions'][0]
                            break
                else:
                    self.api_version = '2018-05-01'
                if not self.api_version:
                    self.fail("Couldn't find api version for {0}/{1}".format(provider, resourceType))
            except Exception as exc:
                self.fail('Failed to obtain API version: {0}'.format(str(exc)))
        self.results['url'] = self.url
        query_parameters = {}
        query_parameters['api-version'] = self.api_version
        header_parameters = {}
        header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        skiptoken = None
        while True:
            if skiptoken:
                query_parameters['skiptoken'] = skiptoken
            response = self.mgmt_client.query(self.url, self.method, query_parameters, header_parameters, None, [200, 404], 0, 0)
            try:
                response = json.loads(response.body())
                if isinstance(response, dict):
                    if response.get('value'):
                        self.results['response'] = self.results['response'] + response['value']
                        skiptoken = response.get('nextLink')
                    else:
                        self.results['response'] = self.results['response'] + [response]
            except Exception as e:
                self.fail('Failed to parse response: ' + str(e))
            if not skiptoken:
                break
        return self.results