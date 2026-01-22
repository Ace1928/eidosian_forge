import sys
import threading
from .__wrapt__ import ObjectProxy
class _ImportHookLoader:

    def load_module(self, fullname):
        module = sys.modules[fullname]
        notify_module_loaded(module)
        return module