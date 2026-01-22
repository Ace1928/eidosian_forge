import sys
import types
import toolz
from importlib import import_module
from importlib.machinery import ModuleSpec
def _load_toolz(self, fullname):
    rv = {}
    package, dot, submodules = fullname.partition('.')
    try:
        module_name = ''.join(['cytoolz', dot, submodules])
        rv['cytoolz'] = import_module(module_name)
    except ImportError:
        pass
    try:
        module_name = ''.join(['toolz', dot, submodules])
        rv['toolz'] = import_module(module_name)
    except ImportError:
        pass
    if not rv:
        raise ImportError(fullname)
    return rv