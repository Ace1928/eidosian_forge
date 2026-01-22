import collections
import importlib
import os
import pkgutil
def _import_modules(module_names):
    imported_modules = []
    for module_name in module_names:
        full_module_path = '.'.join(__name__.split('.')[:-1] + [module_name])
        module = importlib.import_module(full_module_path)
        if not hasattr(module, LIST_OPTS_FUNC_NAME):
            raise Exception("The module '%s' should have a '%s' function which returns the config options." % (full_module_path, LIST_OPTS_FUNC_NAME))
        else:
            imported_modules.append(module)
    return imported_modules