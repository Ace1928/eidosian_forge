from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMFunctionApp(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True, aliases=['resource_group_name']), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), storage_account=dict(type='str', aliases=['storage', 'storage_account_name']), app_settings=dict(type='dict'), plan=dict(type='raw'), container_settings=dict(type='dict', options=container_settings_spec))
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.storage_account = None
        self.app_settings = None
        self.plan = None
        self.container_settings = None
        required_if = [('state', 'present', ['storage_account'])]
        super(AzureRMFunctionApp, self).__init__(self.module_arg_spec, supports_check_mode=True, required_if=required_if)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.app_settings is None:
            self.app_settings = dict()
        try:
            resource_group = self.rm_client.resource_groups.get(self.resource_group)
        except Exception:
            self.fail('Unable to retrieve resource group')
        self.location = self.location or resource_group.location
        try:
            function_app = self.web_client.web_apps.get(resource_group_name=self.resource_group, name=self.name)
            exists = function_app is not None
        except ResourceNotFoundError as exc:
            exists = False
        if self.state == 'absent':
            if exists:
                if self.check_mode:
                    self.results['changed'] = True
                    return self.results
                try:
                    self.web_client.web_apps.delete(resource_group_name=self.resource_group, name=self.name)
                    self.results['changed'] = True
                except Exception as exc:
                    self.fail('Failure while deleting web app: {0}'.format(exc))
            else:
                self.results['changed'] = False
        else:
            kind = 'functionapp'
            linux_fx_version = None
            if self.container_settings and self.container_settings.get('name'):
                kind = 'functionapp,linux,container'
                linux_fx_version = 'DOCKER|'
                if self.container_settings.get('registry_server_url'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_URL'] = 'https://' + self.container_settings['registry_server_url']
                    linux_fx_version += self.container_settings['registry_server_url'] + '/'
                linux_fx_version += self.container_settings['name']
                if self.container_settings.get('registry_server_user'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_USERNAME'] = self.container_settings.get('registry_server_user')
                if self.container_settings.get('registry_server_password'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_PASSWORD'] = self.container_settings.get('registry_server_password')
            if not self.plan and exists:
                self.plan = function_app.server_farm_id
            if not exists:
                function_app = Site(location=self.location, kind=kind, site_config=SiteConfig(app_settings=self.aggregated_app_settings(), scm_type='LocalGit'))
                self.results['changed'] = True
            else:
                self.results['changed'], function_app = self.update(function_app)
            if self.plan:
                if isinstance(self.plan, dict):
                    self.plan = '/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Web/serverfarms/{2}'.format(self.subscription_id, self.plan.get('resource_group', self.resource_group), self.plan.get('name'))
                function_app.server_farm_id = self.plan
            if linux_fx_version:
                function_app.site_config.linux_fx_version = linux_fx_version
            if self.check_mode:
                self.results['state'] = function_app.as_dict()
            elif self.results['changed']:
                try:
                    response = self.web_client.web_apps.begin_create_or_update(resource_group_name=self.resource_group, name=self.name, site_envelope=function_app)
                    new_function_app = self.get_poller_result(response)
                    self.results['state'] = new_function_app.as_dict()
                except Exception as exc:
                    self.fail('Error creating or updating web app: {0}'.format(exc))
        return self.results

    def update(self, source_function_app):
        """Update the Site object if there are any changes"""
        source_app_settings = self.web_client.web_apps.list_application_settings(resource_group_name=self.resource_group, name=self.name)
        changed, target_app_settings = self.update_app_settings(source_app_settings.properties)
        source_function_app.site_config = SiteConfig(app_settings=target_app_settings, scm_type='LocalGit')
        return (changed, source_function_app)

    def update_app_settings(self, source_app_settings):
        """Update app settings"""
        target_app_settings = self.aggregated_app_settings()
        target_app_settings_dict = dict([(i.name, i.value) for i in target_app_settings])
        return (target_app_settings_dict != source_app_settings, target_app_settings)

    def necessary_functionapp_settings(self):
        """Construct the necessary app settings required for an Azure Function App"""
        function_app_settings = []
        if self.container_settings is None:
            for key in ['AzureWebJobsStorage', 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING', 'AzureWebJobsDashboard']:
                function_app_settings.append(NameValuePair(name=key, value=self.storage_connection_string))
            function_app_settings.append(NameValuePair(name='FUNCTIONS_EXTENSION_VERSION', value='~1'))
            function_app_settings.append(NameValuePair(name='WEBSITE_NODE_DEFAULT_VERSION', value='6.5.0'))
            function_app_settings.append(NameValuePair(name='WEBSITE_CONTENTSHARE', value=self.name))
        else:
            function_app_settings.append(NameValuePair(name='FUNCTIONS_EXTENSION_VERSION', value='~2'))
            function_app_settings.append(NameValuePair(name='WEBSITES_ENABLE_APP_SERVICE_STORAGE', value=False))
            function_app_settings.append(NameValuePair(name='AzureWebJobsStorage', value=self.storage_connection_string))
        return function_app_settings

    def aggregated_app_settings(self):
        """Combine both system and user app settings"""
        function_app_settings = self.necessary_functionapp_settings()
        for app_setting_key in self.app_settings:
            found_setting = None
            for s in function_app_settings:
                if s.name == app_setting_key:
                    found_setting = s
                    break
            if found_setting:
                found_setting.value = self.app_settings[app_setting_key]
            else:
                function_app_settings.append(NameValuePair(name=app_setting_key, value=self.app_settings[app_setting_key]))
        return function_app_settings

    @property
    def storage_connection_string(self):
        """Construct the storage account connection string"""
        return 'DefaultEndpointsProtocol=https;AccountName={0};AccountKey={1}'.format(self.storage_account, self.storage_key)

    @property
    def storage_key(self):
        """Retrieve the storage account key"""
        return self.storage_client.storage_accounts.list_keys(resource_group_name=self.resource_group, account_name=self.storage_account).keys[0].value