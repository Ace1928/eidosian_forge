from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def verify_array(self, verify_interface, **kwargs):
    if verify_interface is None:
        return list()
    if isinstance(verify_interface, list):
        if len(verify_interface) == 0:
            return list()
        if verify_interface[0] is None:
            return list()
    return verify_interface