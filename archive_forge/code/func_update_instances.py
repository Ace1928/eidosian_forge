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
def update_instances(old, new):
    """Use garbage collector to find all instances that refer to the old
    class definition and update their __class__ to point to the new class
    definition"""
    refs = gc.get_referrers(old)
    for ref in refs:
        if type(ref) is old:
            object.__setattr__(ref, '__class__', new)