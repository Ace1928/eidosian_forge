import glob
import inspect
import os
import re
import sys
from importlib import import_module
from importlib.machinery import all_suffixes
from time import time
from zipimport import zipimporter
from .completer import expand_user, compress_user
from .error import TryNext
from ..utils._process_common import arg_split
from IPython import get_ipython
from typing import List
import_re = re.compile(r'(?P<name>[^\W\d]\w*?)'
def try_import(mod: str, only_modules=False) -> List[str]:
    """
    Try to import given module and return list of potential completions.
    """
    mod = mod.rstrip('.')
    try:
        m = import_module(mod)
    except:
        return []
    m_is_init = '__init__' in (getattr(m, '__file__', '') or '')
    completions = []
    if not hasattr(m, '__file__') or not only_modules or m_is_init:
        completions.extend([attr for attr in dir(m) if is_importable(m, attr, only_modules)])
    m_all = getattr(m, '__all__', [])
    if only_modules:
        completions.extend((attr for attr in m_all if is_possible_submodule(m, attr)))
    else:
        completions.extend(m_all)
    if m_is_init:
        file_ = m.__file__
        file_path = os.path.dirname(file_)
        if file_path is not None:
            completions.extend(module_list(file_path))
    completions_set = {c for c in completions if isinstance(c, str)}
    completions_set.discard('__init__')
    return list(completions_set)