from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitlabInstanceVariables(object):

    def __init__(self, module, gitlab_instance):
        self.instance = gitlab_instance
        self._module = module

    def list_all_instance_variables(self):
        return list(self.instance.variables.list(**list_all_kwargs))

    def create_variable(self, var_obj):
        if self._module.check_mode:
            return True
        var = {'key': var_obj.get('key'), 'value': var_obj.get('value'), 'masked': var_obj.get('masked'), 'protected': var_obj.get('protected'), 'variable_type': var_obj.get('variable_type')}
        self.instance.variables.create(var)
        return True

    def update_variable(self, var_obj):
        if self._module.check_mode:
            return True
        self.delete_variable(var_obj)
        self.create_variable(var_obj)
        return True

    def delete_variable(self, var_obj):
        if self._module.check_mode:
            return True
        self.instance.variables.delete(var_obj.get('key'))
        return True