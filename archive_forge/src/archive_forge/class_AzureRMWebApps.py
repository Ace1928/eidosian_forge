from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMWebApps(AzureRMModuleBase):
    """Configuration class for an Azure RM Web App resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), plan=dict(type='raw'), frameworks=dict(type='list', elements='dict', options=framework_spec), container_settings=dict(type='dict', options=container_settings_spec), scm_type=dict(type='str'), always_on=dict(type='bool'), min_tls_version=dict(type='str', choices=['1.0', '1.1', '1.2']), ftps_state=dict(type='str', choices=['AllAllowed', 'FtpsOnly', 'Disabled']), deployment_source=dict(type='dict', options=deployment_source_spec), startup_file=dict(type='str'), client_affinity_enabled=dict(type='bool', default=True), https_only=dict(type='bool'), app_settings=dict(type='dict'), purge_app_settings=dict(type='bool', default=False), app_state=dict(type='str', choices=['started', 'stopped', 'restarted'], default='started'), state=dict(type='str', default='present', choices=['present', 'absent']))
        mutually_exclusive = [['container_settings', 'frameworks']]
        self.resource_group = None
        self.name = None
        self.location = None
        self.client_affinity_enabled = True
        self.https_only = None
        self.tags = None
        self.site_config = dict()
        self.app_settings = dict()
        self.app_settings_strDic = None
        self.plan = None
        self.deployment_source = dict()
        self.site = None
        self.container_settings = None
        self.purge_app_settings = False
        self.app_state = 'started'
        self.results = dict(changed=False, id=None)
        self.state = None
        self.to_do = []
        self.frameworks = None
        self.site_config_updatable_properties = ['net_framework_version', 'java_version', 'php_version', 'python_version', 'scm_type', 'always_on', 'min_tls_version', 'ftps_state']
        self.updatable_properties = ['client_affinity_enabled', 'https_only']
        self.supported_linux_frameworks = ['ruby', 'php', 'python', 'dotnetcore', 'node', 'java']
        self.supported_windows_frameworks = ['net_framework', 'php', 'python', 'node', 'java']
        super(AzureRMWebApps, self).__init__(derived_arg_spec=self.module_arg_spec, mutually_exclusive=mutually_exclusive, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key in ['scm_type', 'always_on', 'min_tls_version', 'ftps_state']:
                    self.site_config[key] = kwargs[key]
        old_response = None
        response = None
        to_be_updated = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        old_response = self.get_webapp()
        if old_response:
            self.results['id'] = old_response['id']
        if self.state == 'present':
            if not self.plan and (not old_response):
                self.fail('Please specify plan for newly created web app.')
            if not self.plan:
                self.plan = old_response['server_farm_id']
            self.plan = self.parse_resource_to_dict(self.plan)
            is_linux = False
            old_plan = self.get_app_service_plan()
            if old_plan:
                is_linux = old_plan['reserved']
            else:
                is_linux = self.plan['is_linux'] if 'is_linux' in self.plan else False
            if self.frameworks:
                if len(self.frameworks) > 1 and any((f['name'] == 'java' for f in self.frameworks)):
                    self.fail('Java is mutually exclusive with other frameworks.')
                if is_linux:
                    if len(self.frameworks) != 1:
                        self.fail('Can specify one framework only for Linux web app.')
                    if self.frameworks[0]['name'] not in self.supported_linux_frameworks:
                        self.fail('Unsupported framework {0} for Linux web app.'.format(self.frameworks[0]['name']))
                    self.site_config['linux_fx_version'] = (self.frameworks[0]['name'] + '|' + self.frameworks[0]['version']).upper()
                    if self.frameworks[0]['name'] == 'java':
                        if self.frameworks[0]['version'] != '8':
                            self.fail('Linux web app only supports java 8.')
                        if self.frameworks[0]['settings'] and self.frameworks[0]['settings']['java_container'].lower() != 'tomcat':
                            self.fail('Linux web app only supports tomcat container.')
                        if self.frameworks[0]['settings'] and self.frameworks[0]['settings']['java_container'].lower() == 'tomcat':
                            self.site_config['linux_fx_version'] = 'TOMCAT|' + self.frameworks[0]['settings']['java_container_version'] + '-jre8'
                        else:
                            self.site_config['linux_fx_version'] = 'JAVA|8-jre8'
                else:
                    for fx in self.frameworks:
                        if fx.get('name') not in self.supported_windows_frameworks:
                            self.fail('Unsupported framework {0} for Windows web app.'.format(fx.get('name')))
                        else:
                            self.site_config[fx.get('name') + '_version'] = fx.get('version')
                        if 'settings' in fx and fx['settings'] is not None:
                            for key, value in fx['settings'].items():
                                self.site_config[key] = value
            if not self.app_settings:
                self.app_settings = dict()
            if self.container_settings:
                linux_fx_version = 'DOCKER|'
                if self.container_settings.get('registry_server_url'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_URL'] = 'https://' + self.container_settings['registry_server_url']
                    linux_fx_version += self.container_settings['registry_server_url'] + '/'
                linux_fx_version += self.container_settings['name']
                if self.container_settings['name'].startswith('COMPOSE|') or self.container_settings['name'].startswith('KUBE|'):
                    linux_fx_version = self.container_settings['name']
                self.site_config['linux_fx_version'] = linux_fx_version
                if self.container_settings.get('registry_server_user'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_USERNAME'] = self.container_settings['registry_server_user']
                if self.container_settings.get('registry_server_password'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_PASSWORD'] = self.container_settings['registry_server_password']
            self.site = Site(location=self.location, site_config=self.site_config)
            if self.https_only is not None:
                self.site.https_only = self.https_only
            self.site.client_affinity_enabled = self.client_affinity_enabled
            if not old_response:
                self.log("Web App instance doesn't exist")
                to_be_updated = True
                self.to_do.append(Actions.CreateOrUpdate)
                self.site.tags = self.tags
                if not self.plan:
                    self.fail('Please specify app service plan in plan parameter.')
                if not old_plan:
                    if not self.plan.get('name') or not self.plan.get('sku'):
                        self.fail('Please specify name, is_linux, sku in plan')
                    if 'location' not in self.plan:
                        plan_resource_group = self.get_resource_group(self.plan['resource_group'])
                        self.plan['location'] = plan_resource_group.location
                    old_plan = self.create_app_service_plan()
                self.site.server_farm_id = old_plan['id']
                if old_plan['is_linux']:
                    if hasattr(self, 'startup_file'):
                        self.site_config['app_command_line'] = self.startup_file
                if self.app_settings:
                    app_settings = []
                    for key in self.app_settings.keys():
                        app_settings.append(NameValuePair(name=key, value=self.app_settings[key]))
                    self.site_config['app_settings'] = app_settings
            else:
                self.log('Web App instance already exists')
                self.log('Result: {0}'.format(old_response))
                update_tags, self.site.tags = self.update_tags(old_response.get('tags', None))
                if update_tags:
                    to_be_updated = True
                if self.is_updatable_property_changed(old_response):
                    to_be_updated = True
                    self.to_do.append(Actions.CreateOrUpdate)
                old_config = self.get_webapp_configuration()
                if self.is_site_config_changed(old_config):
                    to_be_updated = True
                    self.to_do.append(Actions.CreateOrUpdate)
                if old_config.linux_fx_version != self.site_config.get('linux_fx_version', ''):
                    to_be_updated = True
                    self.to_do.append(Actions.CreateOrUpdate)
                self.app_settings_strDic = self.list_app_settings()
                if self.purge_app_settings:
                    to_be_updated = True
                    self.app_settings_strDic = dict()
                    self.to_do.append(Actions.UpdateAppSettings)
                if self.purge_app_settings or self.is_app_settings_changed():
                    to_be_updated = True
                    self.to_do.append(Actions.UpdateAppSettings)
                    if self.app_settings:
                        for key in self.app_settings.keys():
                            self.app_settings_strDic[key] = self.app_settings[key]
        elif self.state == 'absent':
            if old_response:
                self.log('Delete Web App instance')
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                self.delete_webapp()
                self.log('Web App instance deleted')
            else:
                self.log('Web app {0} not exists.'.format(self.name))
        if to_be_updated:
            self.log('Need to Create/Update web app')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            if Actions.CreateOrUpdate in self.to_do:
                response = self.create_update_webapp()
                self.results['id'] = response['id']
            if Actions.UpdateAppSettings in self.to_do:
                update_response = self.update_app_settings()
                self.results['id'] = update_response.id
        webapp = None
        if old_response:
            webapp = old_response
        if response:
            webapp = response
        if webapp:
            if webapp['state'] != 'Stopped' and self.app_state == 'stopped' or (webapp['state'] != 'Running' and self.app_state == 'started') or self.app_state == 'restarted':
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                self.set_webapp_state(self.app_state)
        return self.results

    def is_updatable_property_changed(self, existing_webapp):
        for property_name in self.updatable_properties:
            if hasattr(self, property_name) and getattr(self, property_name) is not None and (getattr(self, property_name) != existing_webapp.get(property_name, None)):
                return True
        return False

    def is_site_config_changed(self, existing_config):
        for updatable_property in self.site_config_updatable_properties:
            if self.site_config.get(updatable_property):
                if not getattr(existing_config, updatable_property) or str(getattr(existing_config, updatable_property)).upper() != str(self.site_config.get(updatable_property)).upper():
                    return True
        return False

    def is_app_settings_changed(self):
        if self.app_settings:
            if self.app_settings_strDic:
                for key in self.app_settings.keys():
                    if self.app_settings[key] != self.app_settings_strDic.get(key, None):
                        return True
            else:
                return True
        return False

    def is_deployment_source_changed(self, existing_webapp):
        if self.deployment_source:
            if self.deployment_source.get('url') and self.deployment_source['url'] != existing_webapp.get('site_source_control')['url']:
                return True
            if self.deployment_source.get('branch') and self.deployment_source['branch'] != existing_webapp.get('site_source_control')['branch']:
                return True
        return False

    def create_update_webapp(self):
        """
        Creates or updates Web App with the specified configuration.

        :return: deserialized Web App instance state dictionary
        """
        self.log('Creating / Updating the Web App instance {0}'.format(self.name))
        try:
            response = self.web_client.web_apps.begin_create_or_update(resource_group_name=self.resource_group, name=self.name, site_envelope=self.site)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Web App instance.')
            self.fail('Error creating the Web App instance: {0}'.format(str(exc)))
        return webapp_to_dict(response)

    def delete_webapp(self):
        """
        Deletes specified Web App instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Web App instance {0}'.format(self.name))
        try:
            self.web_client.web_apps.delete(resource_group_name=self.resource_group, name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the Web App instance.')
            self.fail('Error deleting the Web App instance: {0}'.format(str(e)))
        return True

    def get_webapp(self):
        """
        Gets the properties of the specified Web App.

        :return: deserialized Web App instance state dictionary
        """
        self.log('Checking if the Web App instance {0} is present'.format(self.name))
        response = None
        try:
            response = self.web_client.web_apps.get(resource_group_name=self.resource_group, name=self.name)
            if response is not None:
                self.log('Response : {0}'.format(response))
                self.log('Web App instance : {0} found'.format(response.name))
                return webapp_to_dict(response)
        except ResourceNotFoundError:
            pass
        self.log("Didn't find web app {0} in resource group {1}".format(self.name, self.resource_group))
        return False

    def get_app_service_plan(self):
        """
        Gets app service plan
        :return: deserialized app service plan dictionary
        """
        self.log('Get App Service Plan {0}'.format(self.plan['name']))
        try:
            response = self.web_client.app_service_plans.get(resource_group_name=self.plan['resource_group'], name=self.plan['name'])
            if response is not None:
                self.log('Response : {0}'.format(response))
                self.log('App Service Plan : {0} found'.format(response.name))
                return appserviceplan_to_dict(response)
        except ResourceNotFoundError:
            pass
        self.log("Didn't find app service plan {0} in resource group {1}".format(self.plan['name'], self.plan['resource_group']))
        return False

    def create_app_service_plan(self):
        """
        Creates app service plan
        :return: deserialized app service plan dictionary
        """
        self.log('Create App Service Plan {0}'.format(self.plan['name']))
        try:
            sku = _normalize_sku(self.plan['sku'])
            sku_def = SkuDescription(tier=get_sku_name(sku), name=sku, capacity=self.plan.get('number_of_workers', None))
            plan_def = AppServicePlan(location=self.plan['location'], app_service_plan_name=self.plan['name'], sku=sku_def, reserved=self.plan.get('is_linux', None))
            poller = self.web_client.app_service_plans.begin_create_or_update(resource_group_name=self.plan['resource_group'], name=self.plan['name'], app_service_plan=plan_def)
            if isinstance(poller, LROPoller):
                response = self.get_poller_result(poller)
            self.log('Response : {0}'.format(response))
            return appserviceplan_to_dict(response)
        except Exception as ex:
            self.fail('Failed to create app service plan {0} in resource group {1}: {2}'.format(self.plan['name'], self.plan['resource_group'], str(ex)))

    def list_app_settings(self):
        """
        List application settings
        :return: deserialized list response
        """
        self.log('List application setting')
        try:
            response = self.web_client.web_apps.list_application_settings(resource_group_name=self.resource_group, name=self.name)
            self.log('Response : {0}'.format(response))
            return response.properties
        except Exception as ex:
            self.fail('Failed to list application settings for web app {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))

    def update_app_settings(self):
        """
        Update application settings
        :return: deserialized updating response
        """
        self.log('Update application setting')
        try:
            settings = StringDictionary(properties=self.app_settings_strDic)
            response = self.web_client.web_apps.update_application_settings(resource_group_name=self.resource_group, name=self.name, app_settings=settings)
            self.log('Response : {0}'.format(response))
            return response
        except Exception as ex:
            self.fail('Failed to update application settings for web app {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))

    def create_or_update_source_control(self):
        """
        Update site source control
        :return: deserialized updating response
        """
        self.log('Update site source control')
        if self.deployment_source is None:
            return False
        self.deployment_source['is_manual_integration'] = False
        self.deployment_source['is_mercurial'] = False
        try:
            site_source_control = SiteSourceControl(repo_url=self.deployment_source.get('url'), branch=self.deployment_source.get('branch'))
            response = self.web_client.web_apps.begin_create_or_update_source_control(resource_group_name=self.resource_group, name=self.name, site_source_control=site_source_control)
            self.log('Response : {0}'.format(response))
            return response.as_dict()
        except Exception:
            self.fail('Failed to update site source control for web app {0} in resource group {1}'.format(self.name, self.resource_group))

    def get_webapp_configuration(self):
        """
        Get  web app configuration
        :return: deserialized  web app configuration response
        """
        self.log('Get web app configuration')
        try:
            response = self.web_client.web_apps.get_configuration(resource_group_name=self.resource_group, name=self.name)
            self.log('Response : {0}'.format(response))
            return response
        except ResourceNotFoundError as ex:
            self.log('Failed to get configuration for web app {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
            return False

    def set_webapp_state(self, appstate):
        """
        Start/stop/restart web app
        :return: deserialized updating response
        """
        try:
            if appstate == 'started':
                response = self.web_client.web_apps.start(resource_group_name=self.resource_group, name=self.name)
            elif appstate == 'stopped':
                response = self.web_client.web_apps.stop(resource_group_name=self.resource_group, name=self.name)
            elif appstate == 'restarted':
                response = self.web_client.web_apps.restart(resource_group_name=self.resource_group, name=self.name)
            else:
                self.fail('Invalid web app state {0}'.format(appstate))
            self.log('Response : {0}'.format(response))
            return response
        except Exception as ex:
            request_id = ex.request_id if ex.request_id else ''
            self.log('Failed to {0} web app {1} in resource group {2}, request_id {3} - {4}'.format(appstate, self.name, self.resource_group, request_id, str(ex)))