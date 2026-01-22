from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _size_usable_query(v):
    if v.size == 1:
        return 0
    elif v.size == 2:
        return 2
    return v.size - 2