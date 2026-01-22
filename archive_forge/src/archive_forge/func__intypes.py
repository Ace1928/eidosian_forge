import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _intypes(object):
    """check if object is in the 'types' module"""
    import types
    if type(object) is not type(''):
        object = getname(object, force=True)
    if object == 'ellipsis':
        object = 'EllipsisType'
    return True if hasattr(types, object) else False