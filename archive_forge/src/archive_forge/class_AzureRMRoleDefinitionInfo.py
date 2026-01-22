from __future__ import absolute_import, division, print_function
class AzureRMRoleDefinitionInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(scope=dict(type='str', required=True), role_name=dict(type='str'), id=dict(type='str'), type=dict(type='str', choices=['custom', 'system']))
        self.role_name = None
        self.scope = None
        self.id = None
        self.type = None
        self.results = dict(changed=False)
        self._client = None
        super(AzureRMRoleDefinitionInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        is_old_facts = self.module._name == 'azure_rm_roledefinition_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_roledefinition_facts' module has been renamed to 'azure_rm_roledefinition_info'", version=(2.9,))
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        if self.type:
            self.type = self.get_role_type(self.type)
        self._client = self.get_mgmt_svc_client(AuthorizationManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2018-01-01-preview')
        if self.id:
            self.results['roledefinitions'] = self.get_by_id()
        elif self.role_name:
            self.results['roledefinitions'] = self.get_by_role_name()
        else:
            self.results['roledefinitions'] = self.list()
        return self.results

    def get_role_type(self, role_type):
        if role_type:
            if role_type == 'custom':
                return 'CustomRole'
            else:
                return 'SystemRole'
        return role_type

    def list(self):
        """
        List Role Definition in scope.

        :return: deserialized Role Definition state dictionary
        """
        self.log('List Role Definition in scope {0}'.format(self.scope))
        response = []
        try:
            response = list(self._client.role_definitions.list(scope=self.scope))
            if len(response) > 0:
                self.log('Response : {0}'.format(response))
                roles = []
                if self.type:
                    roles = [r for r in response if r.role_type == self.type]
                else:
                    roles = response
                if len(roles) > 0:
                    return [roledefinition_to_dict(r) for r in roles]
        except Exception as ex:
            self.log("Didn't find role definition in scope {0}".format(self.scope))
        return response

    def get_by_id(self):
        """
        Get Role Definition in scope by id.

        :return: deserialized Role Definition state dictionary
        """
        self.log('Get Role Definition by id {0}'.format(self.id))
        response = None
        try:
            response = self._client.role_definitions.get(scope=self.scope, role_definition_id=self.id)
            if response:
                response = roledefinition_to_dict(response)
                if self.type:
                    if response.role_type == self.type:
                        return [response]
                else:
                    return [response]
        except Exception as ex:
            self.log("Didn't find role definition by id {0}".format(self.id))
        return []

    def get_by_role_name(self):
        """
        Get Role Definition in scope by role name.

        :return: deserialized role definition state dictionary
        """
        self.log('Get Role Definition by name {0}'.format(self.role_name))
        response = []
        try:
            response = self.list()
            if len(response) > 0:
                roles = []
                for r in response:
                    if r['role_name'] == self.role_name:
                        roles.append(r)
                if len(roles) == 1:
                    self.log('Role Definition : {0} found'.format(self.role_name))
                    return roles
                if len(roles) > 1:
                    self.fail('Found multiple Role Definitions with name: {0}'.format(self.role_name))
        except Exception as ex:
            self.log("Didn't find Role Definition by name {0}".format(self.role_name))
        return []