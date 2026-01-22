from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.community.general.plugins.module_utils.mh.base import ModuleHelperBase
from ansible_collections.community.general.plugins.module_utils.mh.deco import module_fails_on_exception
class DependencyCtxMgr(object):

    def __init__(self, name, msg=None):
        self.name = name
        self.msg = msg
        self.has_it = False
        self.exc_type = None
        self.exc_val = None
        self.exc_tb = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.has_it = exc_type is None
        self.exc_type = exc_type
        self.exc_val = exc_val
        self.exc_tb = exc_tb
        return not self.has_it

    @property
    def text(self):
        return self.msg or str(self.exc_val)