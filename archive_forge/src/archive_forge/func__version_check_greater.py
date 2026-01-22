from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
def _version_check_greater(self, greater, lesser, greater_or_equal=False):
    """Determine if first argument is greater than second argument.

        Args:
            greater (str): decimal string
            lesser (str): decimal string
        """
    g_major, g_minor = greater.split('.')
    l_major, l_minor = lesser.split('.')
    g_major = int(g_major)
    g_minor = int(g_minor)
    l_major = int(l_major)
    l_minor = int(l_minor)
    if g_major > l_major:
        return True
    elif greater_or_equal and g_major == l_major and (g_minor >= l_minor):
        return True
    elif g_major == l_major and g_minor > l_minor:
        return True
    return False