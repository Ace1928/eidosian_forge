from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMWebAppInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'), return_publish_profile=dict(type='bool', default=False))
        self.results = dict(changed=False, webapps=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        self.return_publish_profile = False
        self.framework_names = ['net_framework', 'java', 'php', 'node', 'python', 'dotnetcore', 'ruby']
        super(AzureRMWebAppInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_webapp_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_webapp_facts' module has been renamed to 'azure_rm_webapp_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name:
            self.results['webapps'] = self.list_by_name()
        elif self.resource_group:
            self.results['webapps'] = self.list_by_resource_group()
        else:
            self.results['webapps'] = self.list_all()
        return self.results

    def list_by_name(self):
        self.log('Get web app {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.web_client.web_apps.get(resource_group_name=self.resource_group, name=self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            curated_result = self.get_curated_webapp(self.resource_group, self.name, item)
            result = [curated_result]
        return result

    def list_by_resource_group(self):
        self.log('List web apps in resource groups {0}'.format(self.resource_group))
        try:
            response = list(self.web_client.web_apps.list_by_resource_group(resource_group_name=self.resource_group))
        except Exception as exc:
            request_id = exc.request_id if exc.request_id else ''
            self.fail('Error listing web apps in resource groups {0}, request id: {1} - {2}'.format(self.resource_group, request_id, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                curated_output = self.get_curated_webapp(self.resource_group, item.name, item)
                results.append(curated_output)
        return results

    def list_all(self):
        self.log('List web apps in current subscription')
        try:
            response = list(self.web_client.web_apps.list())
        except Exception as exc:
            request_id = exc.request_id if exc.request_id else ''
            self.fail('Error listing web apps, request id {0} - {1}'.format(request_id, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                curated_output = self.get_curated_webapp(item.resource_group, item.name, item)
                results.append(curated_output)
        return results

    def list_webapp_configuration(self, resource_group, name):
        self.log('Get web app {0} configuration'.format(name))
        response = []
        try:
            response = self.web_client.web_apps.get_configuration(resource_group_name=resource_group, name=name)
        except Exception as ex:
            request_id = ex.request_id if ex.request_id else ''
            self.fail('Error getting web app {0} configuration, request id {1} - {2}'.format(name, request_id, str(ex)))
        return response.as_dict()

    def list_webapp_appsettings(self, resource_group, name):
        self.log('Get web app {0} app settings'.format(name))
        response = []
        try:
            response = self.web_client.web_apps.list_application_settings(resource_group_name=resource_group, name=name)
        except Exception as ex:
            request_id = ex.request_id if ex.request_id else ''
            self.fail('Error getting web app {0} app settings, request id {1} - {2}'.format(name, request_id, str(ex)))
        return response.as_dict()

    def get_publish_credentials(self, resource_group, name):
        self.log('Get web app {0} publish credentials'.format(name))
        try:
            poller = self.web_client.web_apps.begin_list_publishing_credentials(resource_group_name=resource_group, name=name)
            if isinstance(poller, LROPoller):
                response = self.get_poller_result(poller)
        except Exception as ex:
            request_id = ex.request_id if ex.request_id else ''
            self.fail('Error getting web app {0} publishing credentials - {1}'.format(request_id, str(ex)))
        return response

    def get_webapp_ftp_publish_url(self, resource_group, name):
        self.log('Get web app {0} app publish profile'.format(name))
        url = None
        try:
            publishing_profile_options = CsmPublishingProfileOptions(format='Ftp')
            content = self.web_client.web_apps.list_publishing_profile_xml_with_secrets(resource_group_name=resource_group, name=name, publishing_profile_options=publishing_profile_options)
            if not content:
                return url
            full_xml = ''
            for f in content:
                full_xml += f.decode()
            profiles = xmltodict.parse(full_xml, xml_attribs=True)['publishData']['publishProfile']
            if not profiles:
                return url
            for profile in profiles:
                if profile['@publishMethod'] == 'FTP':
                    url = profile['@publishUrl']
        except Exception as ex:
            self.fail('Error getting web app {0} app settings - {1}'.format(name, str(ex)))
        return url

    def get_curated_webapp(self, resource_group, name, webapp):
        pip = self.serialize_obj(webapp, AZURE_OBJECT_CLASS)
        try:
            site_config = self.list_webapp_configuration(resource_group, name)
            app_settings = self.list_webapp_appsettings(resource_group, name)
            publish_cred = self.get_publish_credentials(resource_group, name)
            ftp_publish_url = self.get_webapp_ftp_publish_url(resource_group, name)
        except Exception:
            pass
        return self.construct_curated_webapp(webapp=pip, configuration=site_config, app_settings=app_settings, deployment_slot=None, ftp_publish_url=ftp_publish_url, publish_credentials=publish_cred)

    def construct_curated_webapp(self, webapp, configuration=None, app_settings=None, deployment_slot=None, ftp_publish_url=None, publish_credentials=None):
        curated_output = dict()
        curated_output['id'] = webapp['id']
        curated_output['name'] = webapp['name']
        curated_output['resource_group'] = webapp['resource_group']
        curated_output['location'] = webapp['location']
        curated_output['plan'] = webapp['server_farm_id']
        curated_output['tags'] = webapp.get('tags', None)
        curated_output['app_state'] = webapp['state']
        curated_output['availability_state'] = webapp['availability_state']
        curated_output['default_host_name'] = webapp['default_host_name']
        curated_output['host_names'] = webapp['host_names']
        curated_output['enabled'] = webapp['enabled']
        curated_output['enabled_host_names'] = webapp['enabled_host_names']
        curated_output['host_name_ssl_states'] = webapp['host_name_ssl_states']
        curated_output['outbound_ip_addresses'] = webapp['outbound_ip_addresses']
        if configuration:
            curated_output['frameworks'] = []
            for fx_name in self.framework_names:
                fx_version = configuration.get(fx_name + '_version', None)
                if fx_version:
                    fx = {'name': fx_name, 'version': fx_version}
                    if fx_name == 'java':
                        if configuration['java_container'] and configuration['java_container_version']:
                            settings = {'java_container': configuration['java_container'].lower(), 'java_container_version': configuration['java_container_version']}
                            fx['settings'] = settings
                    curated_output['frameworks'].append(fx)
            if configuration.get('linux_fx_version', None):
                tmp = configuration.get('linux_fx_version').split('|')
                if len(tmp) == 2:
                    curated_output['frameworks'].append({'name': tmp[0].lower(), 'version': tmp[1]})
            curated_output['always_on'] = configuration.get('always_on')
            curated_output['ftps_state'] = configuration.get('ftps_state')
            curated_output['min_tls_version'] = configuration.get('min_tls_version')
        if app_settings and app_settings.get('properties', None):
            curated_output['app_settings'] = dict()
            for item in app_settings['properties']:
                curated_output['app_settings'][item] = app_settings['properties'][item]
        if deployment_slot:
            curated_output['deployment_slot'] = deployment_slot
        if ftp_publish_url:
            curated_output['ftp_publish_url'] = ftp_publish_url
        if publish_credentials and self.return_publish_profile:
            curated_output['publishing_username'] = publish_credentials.publishing_user_name
            curated_output['publishing_password'] = publish_credentials.publishing_password
        return curated_output