from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six.moves.collections_abc import MutableMapping
def _camel_to_snake(name, reversible=False):

    def prepend_underscore_and_lower(m):
        return '_' + m.group(0).lower()
    if reversible:
        upper_pattern = '[A-Z]'
    else:
        upper_pattern = '[A-Z]{3,}s$'
    s1 = re.sub(upper_pattern, prepend_underscore_and_lower, name)
    if s1.startswith('_') and (not name.startswith('_')):
        s1 = s1[1:]
    if reversible:
        return s1
    first_cap_pattern = '(.)([A-Z][a-z]+)'
    all_cap_pattern = '([a-z0-9])([A-Z]+)'
    s2 = re.sub(first_cap_pattern, '\\1_\\2', s1)
    return re.sub(all_cap_pattern, '\\1_\\2', s2).lower()