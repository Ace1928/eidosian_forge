from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def update_result(self, resource, result=None):
    if result is None:
        result = dict()
    if resource:
        returns = self.common_returns.copy()
        returns.update(self.returns)
        for search_key, return_key in returns.items():
            if search_key in resource:
                result[return_key] = resource[search_key]
        for search_key, return_key in self.returns_to_int.items():
            if search_key in resource:
                result[return_key] = int(resource[search_key])
    return result