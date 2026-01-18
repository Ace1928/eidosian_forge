from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import integer_types, string_types
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import traceback
def vars_to_variables(vars, module):
    variables = list()
    for item, value in vars.items():
        if isinstance(value, (string_types, integer_types, float)):
            variables.append({'name': item, 'value': str(value), 'masked': False, 'protected': False, 'raw': False, 'variable_type': 'env_var'})
        elif isinstance(value, dict):
            new_item = {'name': item, 'value': value.get('value'), 'masked': value.get('masked'), 'protected': value.get('protected'), 'raw': value.get('raw'), 'variable_type': value.get('variable_type')}
            if value.get('environment_scope'):
                new_item['environment_scope'] = value.get('environment_scope')
            variables.append(new_item)
        else:
            module.fail_json(msg='value must be of type string, integer, float or dict')
    return variables