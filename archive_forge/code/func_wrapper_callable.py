from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible_collections.community.general.plugins.module_utils.mh.exceptions import ModuleHelperException
@wraps(func)
def wrapper_callable(self, *args, **kwargs):
    if self.module.check_mode:
        return callable(self, *args, **kwargs)
    return func(self, *args, **kwargs)