import re
import os
import sys
import warnings
from dill import _dill, Pickler, Unpickler
from ._dill import (
from typing import Optional, Union
import pathlib
import tempfile
def _module_map():
    """get map of imported modules"""
    from collections import defaultdict
    from types import SimpleNamespace
    modmap = SimpleNamespace(by_name=defaultdict(list), by_id=defaultdict(list), top_level={})
    for modname, module in sys.modules.items():
        if modname in ('__main__', '__mp_main__') or not isinstance(module, ModuleType):
            continue
        if '.' not in modname:
            modmap.top_level[id(module)] = modname
        for objname, modobj in module.__dict__.items():
            modmap.by_name[objname].append((modobj, modname))
            modmap.by_id[id(modobj)].append((modobj, objname, modname))
    return modmap