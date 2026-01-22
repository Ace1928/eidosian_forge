from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def _var_matches(self, key, search_pattern):
    if self._pattern_type == 'prefix':
        return key.startswith(search_pattern)
    elif self._pattern_type == 'suffix':
        return key.endswith(search_pattern)
    elif self._pattern_type == 'regex':
        matcher = re.compile(search_pattern)
        return matcher.search(key)
    return False