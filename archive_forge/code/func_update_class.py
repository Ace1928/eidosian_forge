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
def update_class(old, new):
    """Replace stuff in the __dict__ of a class, and upgrade
    method code objects, and add new methods, if any"""
    for key in list(old.__dict__.keys()):
        old_obj = getattr(old, key)
        try:
            new_obj = getattr(new, key)
            if (old_obj == new_obj) is True:
                continue
        except AttributeError:
            try:
                delattr(old, key)
            except (AttributeError, TypeError):
                pass
            continue
        except ValueError:
            pass
        if update_generic(old_obj, new_obj):
            continue
        try:
            setattr(old, key, getattr(new, key))
        except (AttributeError, TypeError):
            pass
    for key in list(new.__dict__.keys()):
        if key not in list(old.__dict__.keys()):
            try:
                setattr(old, key, getattr(new, key))
            except (AttributeError, TypeError):
                pass
    update_instances(old, new)