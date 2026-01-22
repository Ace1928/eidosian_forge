from functools import cmp_to_key
import ansible.module_utils.common.warnings as ansible_warnings
from ansible.module_utils._text import to_text
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import string_types
def _canonify_policy_dict_item(item, key):
    """
    Converts special cases where there are multiple ways to write the same thing into a single form
    """
    if key in ['NotPrincipal', 'Principal']:
        if item == '*':
            return {'AWS': '*'}
    return item