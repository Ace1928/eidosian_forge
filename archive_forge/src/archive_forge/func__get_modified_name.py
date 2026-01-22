from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _get_modified_name(variable_name):
    modified_name = variable_name
    for special_char in ['-', ' ', '.', '(', '+']:
        modified_name = modified_name.replace(special_char, '_')
    return modified_name