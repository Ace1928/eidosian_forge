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
def append_obj(module, d, name, obj, autoload=False):
    in_module = hasattr(obj, '__module__') and obj.__module__ == module.__name__
    if autoload:
        if not in_module and name in mod_attrs:
            return False
    elif not in_module:
        return False
    key = (module.__name__, name)
    try:
        d.setdefault(key, []).append(weakref.ref(obj))
    except TypeError:
        pass
    return True