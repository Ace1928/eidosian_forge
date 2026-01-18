from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible_collections.community.general.plugins.module_utils.mh.exceptions import ModuleHelperException
@wraps(func)
def wrapper_value(self, *args, **kwargs):
    if self.module.check_mode:
        return value
    return func(self, *args, **kwargs)