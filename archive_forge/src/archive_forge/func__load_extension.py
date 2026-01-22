import os
import os.path
import sys
from importlib import import_module, reload
from traitlets.config.configurable import Configurable
from IPython.utils.path import ensure_dir_exists
from traitlets import Instance
def _load_extension(self, module_str: str):
    if module_str in self.loaded:
        return 'already loaded'
    assert self.shell is not None
    with self.shell.builtin_trap:
        if module_str not in sys.modules:
            mod = import_module(module_str)
        mod = sys.modules[module_str]
        if self._call_load_ipython_extension(mod):
            self.loaded.add(module_str)
        else:
            return 'no load function'