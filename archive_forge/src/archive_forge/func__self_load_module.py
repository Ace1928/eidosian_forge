import sys
import threading
from .__wrapt__ import ObjectProxy
def _self_load_module(self, fullname):
    module = self.__wrapped__.load_module(fullname)
    self._self_set_loader(module)
    notify_module_loaded(module)
    return module