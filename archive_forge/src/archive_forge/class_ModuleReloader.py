imported with ``from foo import ...`` was also updated.
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
import os
import sys
import traceback
import types
import weakref
import gc
import logging
from importlib import import_module, reload
from importlib.util import source_from_cache
class ModuleReloader:
    enabled = False
    'Whether this reloader is enabled'
    check_all = True
    "Autoreload all modules, not just those listed in 'modules'"
    autoload_obj = False
    'Autoreload all modules AND autoload all new objects'

    def __init__(self, shell=None):
        self.failed = {}
        self.modules = {}
        self.skip_modules = {}
        self.old_objects = {}
        self.modules_mtimes = {}
        self.shell = shell
        self._report = lambda msg: None
        self.check(check_all=True, do_reload=False)
        self.hide_errors = False

    def mark_module_skipped(self, module_name):
        """Skip reloading the named module in the future"""
        try:
            del self.modules[module_name]
        except KeyError:
            pass
        self.skip_modules[module_name] = True

    def mark_module_reloadable(self, module_name):
        """Reload the named module in the future (if it is imported)"""
        try:
            del self.skip_modules[module_name]
        except KeyError:
            pass
        self.modules[module_name] = True

    def aimport_module(self, module_name):
        """Import a module, and mark it reloadable

        Returns
        -------
        top_module : module
            The imported module if it is top-level, or the top-level
        top_name : module
            Name of top_module

        """
        self.mark_module_reloadable(module_name)
        import_module(module_name)
        top_name = module_name.split('.')[0]
        top_module = sys.modules[top_name]
        return (top_module, top_name)

    def filename_and_mtime(self, module):
        if not hasattr(module, '__file__') or module.__file__ is None:
            return (None, None)
        if getattr(module, '__name__', None) in [None, '__mp_main__', '__main__']:
            return (None, None)
        filename = module.__file__
        path, ext = os.path.splitext(filename)
        if ext.lower() == '.py':
            py_filename = filename
        else:
            try:
                py_filename = source_from_cache(filename)
            except ValueError:
                return (None, None)
        try:
            pymtime = os.stat(py_filename).st_mtime
        except OSError:
            return (None, None)
        return (py_filename, pymtime)

    def check(self, check_all=False, do_reload=True):
        """Check whether some modules need to be reloaded."""
        if not self.enabled and (not check_all):
            return
        if check_all or self.check_all:
            modules = list(sys.modules.keys())
        else:
            modules = list(self.modules.keys())
        for modname in modules:
            m = sys.modules.get(modname, None)
            if modname in self.skip_modules:
                continue
            py_filename, pymtime = self.filename_and_mtime(m)
            if py_filename is None:
                continue
            try:
                if pymtime <= self.modules_mtimes[modname]:
                    continue
            except KeyError:
                self.modules_mtimes[modname] = pymtime
                continue
            else:
                if self.failed.get(py_filename, None) == pymtime:
                    continue
            self.modules_mtimes[modname] = pymtime
            if do_reload:
                self._report(f"Reloading '{modname}'.")
                try:
                    if self.autoload_obj:
                        superreload(m, reload, self.old_objects, self.shell)
                    else:
                        superreload(m, reload, self.old_objects)
                    if py_filename in self.failed:
                        del self.failed[py_filename]
                except:
                    if not self.hide_errors:
                        print('[autoreload of {} failed: {}]'.format(modname, traceback.format_exc(10)), file=sys.stderr)
                    self.failed[py_filename] = pymtime