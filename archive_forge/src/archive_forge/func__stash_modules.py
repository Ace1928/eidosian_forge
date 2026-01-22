import re
import os
import sys
import warnings
from dill import _dill, Pickler, Unpickler
from ._dill import (
from typing import Optional, Union
import pathlib
import tempfile
def _stash_modules(main_module):
    modmap = _module_map()
    newmod = ModuleType(main_module.__name__)
    imported = []
    imported_as = []
    imported_top_level = []
    original = {}
    for name, obj in main_module.__dict__.items():
        if obj is main_module:
            original[name] = newmod
        elif obj is main_module.__dict__:
            original[name] = newmod.__dict__
        elif any((obj is singleton for singleton in (None, False, True))) or (isinstance(obj, ModuleType) and _is_builtin_module(obj)):
            original[name] = obj
        else:
            source_module, objname = _lookup_module(modmap, name, obj, main_module)
            if source_module is not None:
                if objname == name:
                    imported.append((source_module, name))
                else:
                    imported_as.append((source_module, objname, name))
            else:
                try:
                    imported_top_level.append((modmap.top_level[id(obj)], name))
                except KeyError:
                    original[name] = obj
    if len(original) < len(main_module.__dict__):
        newmod.__dict__.update(original)
        newmod.__dill_imported = imported
        newmod.__dill_imported_as = imported_as
        newmod.__dill_imported_top_level = imported_top_level
        if getattr(newmod, '__loader__', None) is None and _is_imported_module(main_module):
            newmod.__loader__ = True
        return newmod
    else:
        return main_module