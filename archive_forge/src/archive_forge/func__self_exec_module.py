import sys
import threading
from .__wrapt__ import ObjectProxy
def _self_exec_module(self, module):
    self._self_set_loader(module)
    self.__wrapped__.exec_module(module)
    notify_module_loaded(module)