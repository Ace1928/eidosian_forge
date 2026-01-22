import re
import numpy as np
from pandas import DataFrame
from ...rcparams import rcParams
def _copy_docstring(lib, function):
    """Extract docstring from function."""
    import importlib
    try:
        module = importlib.import_module(lib)
        func = getattr(module, function)
        doc = func.__doc__
    except ImportError:
        doc = f'Failed to import function {function} from {lib}'
    if not isinstance(doc, str):
        doc = ''
    return doc