from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def _merge_dict(self, src, dest, path):
    for key, value in src.items():
        if isinstance(value, dict):
            node = dest.setdefault(key, {})
            self._merge_dict(value, node, path + [key])
        elif isinstance(value, list) and key in dest:
            dest[key] += value
        else:
            if key in dest and dest[key] != value:
                msg = "The key '{0}' with value '{1}' will be overwritten with value '{2}' from '{3}.{0}'".format(key, dest[key], value, '.'.join(path))
                if self._override == 'error':
                    raise AnsibleError(msg)
                if self._override == 'warn':
                    display.warning(msg)
            dest[key] = value
    return dest