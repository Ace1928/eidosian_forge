from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMADApplication(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(tenant=dict(type='str', required=True), app_id=dict(type='str'), display_name=dict(type='str'), app_roles=dict(type='list', elements='dict', options=app_role_spec), available_to_other_tenants=dict(type='bool'), credential_description=dict(type='str'), end_date=dict(type='str'), homepage=dict(type='str'), allow_guests_sign_in=dict(type='bool'), identifier_uris=dict(type='list', elements='str'), key_type=dict(type='str', default='AsymmetricX509Cert', choices=['AsymmetricX509Cert', 'Password', 'Symmetric']), key_usage=dict(type='str', default='Verify', choices=['Sign', 'Verify']), key_value=dict(type='str', no_log=True), native_app=dict(type='bool'), oauth2_allow_implicit_flow=dict(type='bool'), optional_claims=dict(type='list', elements='dict', options=optional_claims_spec), password=dict(type='str', no_log=True), reply_urls=dict(type='list', elements='str'), start_date=dict(type='str'), required_resource_accesses=dict(type='list', elements='dict', options=required_resource_accesses_spec), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.state = None
        self.tenant = None
        self.app_id = None
        self.display_name = None
        self.app_roles = None
        self.available_to_other_tenants = None
        self.credential_description = None
        self.end_date = None
        self.homepage = None
        self.identifier_uris = None
        self.key_type = None
        self.key_usage = None
        self.key_value = None
        self.native_app = None
        self.oauth2_allow_implicit_flow = None
        self.optional_claims = None
        self.password = None
        self.reply_urls = None
        self.start_date = None
        self.required_resource_accesses = None
        self.allow_guests_sign_in = None
        self.results = dict(changed=False)
        super(AzureRMADApplication, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=False, supports_tags=False, is_ad_resource=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        response = self.get_resource()
        if response:
            if self.state == 'present':
                if self.check_update(response):
                    self.update_resource(response)
            elif self.state == 'absent':
                self.delete_resource(response)
        elif self.state == 'present':
            self.create_resource()
        elif self.state == 'absent':
            self.log('try to delete non exist resource')
        return self.results

    def create_resource(self):
        try:
            key_creds, password_creds, required_accesses, app_roles, optional_claims = (None, None, None, None, None)
            if self.native_app:
                if self.identifier_uris:
                    self.fail("'identifier_uris' is not required for creating a native application")
            else:
                password_creds, key_creds = self.build_application_creds(self.password, self.key_value, self.key_type, self.key_usage, self.start_date, self.end_date, self.credential_description)
            if self.required_resource_accesses:
                required_accesses = self.build_application_accesses(self.required_resource_accesses)
            if self.app_roles:
                app_roles = self.build_app_roles(self.app_roles)
            client = self.get_graphrbac_client(self.tenant)
            app_create_param = ApplicationCreateParameters(available_to_other_tenants=self.available_to_other_tenants, display_name=self.display_name, identifier_uris=self.identifier_uris, homepage=self.homepage, reply_urls=self.reply_urls, key_credentials=key_creds, password_credentials=password_creds, oauth2_allow_implicit_flow=self.oauth2_allow_implicit_flow, required_resource_access=required_accesses, app_roles=app_roles, allow_guests_sign_in=self.allow_guests_sign_in, optional_claims=self.optional_claims)
            response = client.applications.create(app_create_param)
            self.results['changed'] = True
            self.results.update(self.to_dict(response))
            return response
        except GraphErrorException as ge:
            self.fail('Error creating application, display_name {0} - {1}'.format(self.display_name, str(ge)))

    def update_resource(self, old_response):
        try:
            client = self.get_graphrbac_client(self.tenant)
            key_creds, password_creds, required_accesses, app_roles, optional_claims = (None, None, None, None, None)
            if self.native_app:
                if self.identifier_uris:
                    self.fail("'identifier_uris' is not required for creating a native application")
            else:
                password_creds, key_creds = self.build_application_creds(self.password, self.key_value, self.key_type, self.key_usage, self.start_date, self.end_date, self.credential_description)
            if self.required_resource_accesses:
                required_accesses = self.build_application_accesses(self.required_resource_accesses)
            if self.app_roles:
                app_roles = self.build_app_roles(self.app_roles)
            app_update_param = ApplicationUpdateParameters(available_to_other_tenants=self.available_to_other_tenants, display_name=self.display_name, identifier_uris=self.identifier_uris, homepage=self.homepage, reply_urls=self.reply_urls, key_credentials=key_creds, password_credentials=password_creds, oauth2_allow_implicit_flow=self.oauth2_allow_implicit_flow, required_resource_access=required_accesses, allow_guests_sign_in=self.allow_guests_sign_in, app_roles=app_roles, optional_claims=self.optional_claims)
            client.applications.patch(old_response['object_id'], app_update_param)
            self.results['changed'] = True
            self.results.update(self.get_resource())
        except GraphErrorException as ge:
            self.fail('Error updating the application app_id {0} - {1}'.format(self.app_id, str(ge)))

    def delete_resource(self, response):
        try:
            client = self.get_graphrbac_client(self.tenant)
            client.applications.delete(response.get('object_id'))
            self.results['changed'] = True
            return True
        except GraphErrorException as ge:
            self.fail('Error deleting application app_id {0} display_name {1} - {2}'.format(self.app_id, self.display_name, str(ge)))

    def get_resource(self):
        try:
            client = self.get_graphrbac_client(self.tenant)
            existing_apps = []
            if self.app_id:
                existing_apps = list(client.applications.list(filter="appId eq '{0}'".format(self.app_id)))
            if not existing_apps:
                return False
            result = existing_apps[0]
            return self.to_dict(result)
        except GraphErrorException as ge:
            self.log('Did not find the graph instance instance {0} - {1}'.format(self.app_id, str(ge)))
            return False

    def check_update(self, response):
        for key in list(self.module_arg_spec.keys()):
            attr = getattr(self, key)
            if attr and key in response:
                if response and attr != response[key] or response[key] is None:
                    return True
        return False

    def to_dict(self, object):
        app_roles = [{'id': app_role.id, 'display_name': app_role.display_name, 'is_enabled': app_role.is_enabled, 'value': app_role.value, 'description': app_role.description} for app_role in object.app_roles]
        return dict(app_id=object.app_id, object_id=object.object_id, display_name=object.display_name, app_roles=app_roles, available_to_other_tenants=object.available_to_other_tenants, homepage=object.homepage, identifier_uris=object.identifier_uris, oauth2_allow_implicit_flow=object.oauth2_allow_implicit_flow, optional_claims=object.optional_claims, allow_guests_sign_in=object.allow_guests_sign_in, reply_urls=object.reply_urls)

    def build_application_creds(self, password=None, key_value=None, key_type=None, key_usage=None, start_date=None, end_date=None, key_description=None):
        if password and key_value:
            self.fail('specify either password or key_value, but not both.')
        if not start_date:
            start_date = datetime.datetime.utcnow()
        elif isinstance(start_date, str):
            start_date = dateutil.parser.parse(start_date)
        if not end_date:
            end_date = start_date + relativedelta(years=1) - relativedelta(hours=24)
        elif isinstance(end_date, str):
            end_date = dateutil.parser.parse(end_date)
        custom_key_id = None
        if key_description and password:
            custom_key_id = self.encode_custom_key_description(key_description)
        key_type = key_type or 'AsymmetricX509Cert'
        key_usage = key_usage or 'Verify'
        password_creds = None
        key_creds = None
        if password:
            password_creds = [PasswordCredential(start_date=start_date, end_date=end_date, key_id=str(self.gen_guid()), value=password, custom_key_identifier=custom_key_id)]
        elif key_value:
            key_creds = [KeyCredential(start_date=start_date, end_date=end_date, key_id=str(self.gen_guid()), value=key_value, usage=key_usage, type=key_type, custom_key_identifier=custom_key_id)]
        return (password_creds, key_creds)

    def encode_custom_key_description(self, key_description):
        return key_description.encode('utf-16')

    def gen_guid(self):
        return uuid.uuid4()

    def build_application_accesses(self, required_resource_accesses):
        if not required_resource_accesses:
            return None
        required_accesses = []
        if isinstance(required_resource_accesses, dict):
            self.log('Getting "requiredResourceAccess" from a full manifest')
            required_resource_accesses = required_resource_accesses.get('required_resource_access', [])
        for x in required_resource_accesses:
            accesses = [ResourceAccess(id=y['id'], type=y['type']) for y in x['resource_access']]
            required_accesses.append(RequiredResourceAccess(resource_app_id=x['resource_app_id'], resource_access=accesses))
        return required_accesses

    def build_app_roles(self, app_roles):
        if not app_roles:
            return None
        result = []
        if isinstance(app_roles, dict):
            self.log('Getting "appRoles" from a full manifest')
            app_roles = app_roles.get('appRoles', [])
        for x in app_roles:
            role = AppRole(id=x.get('id', None) or self.gen_guid(), allowed_member_types=x.get('allowed_member_types', None), description=x.get('description', None), display_name=x.get('display_name', None), is_enabled=x.get('is_enabled', None), value=x.get('value', None))
            result.append(role)
        return result