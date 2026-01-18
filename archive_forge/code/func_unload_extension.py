import os
import os.path
import sys
from importlib import import_module, reload
from traitlets.config.configurable import Configurable
from IPython.utils.path import ensure_dir_exists
from traitlets import Instance
def unload_extension(self, module_str: str):
    """Unload an IPython extension by its module name.

        This function looks up the extension's name in ``sys.modules`` and
        simply calls ``mod.unload_ipython_extension(self)``.

        Returns the string "no unload function" if the extension doesn't define
        a function to unload itself, "not loaded" if the extension isn't loaded,
        otherwise None.
        """
    if BUILTINS_EXTS.get(module_str, False) is True:
        module_str = 'IPython.extensions.' + module_str
    if module_str not in self.loaded:
        return 'not loaded'
    if module_str in sys.modules:
        mod = sys.modules[module_str]
        if self._call_unload_ipython_extension(mod):
            self.loaded.discard(module_str)
        else:
            return 'no unload function'