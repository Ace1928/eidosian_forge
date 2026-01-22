import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _namespace(obj):
    """_namespace(obj); return namespace hierarchy (as a list of names)
    for the given object.  For an instance, find the class hierarchy.

    For example:

    >>> from functools import partial
    >>> p = partial(int, base=2)
    >>> _namespace(p)
    ['functools', 'partial']
    """
    try:
        module = qual = str(getmodule(obj)).split()[1].strip('>').strip('"').strip("'")
        qual = qual.split('.')
        if ismodule(obj):
            return qual
        name = getname(obj) or obj.__name__
        if module in ['builtins', '__builtin__']:
            if _intypes(name):
                return ['types'] + [name]
        return qual + [name]
    except Exception:
        pass
    if str(obj) in ['inf', 'nan', 'Inf', 'NaN']:
        return ['numpy'] + [str(obj)]
    module = getattr(obj.__class__, '__module__', None)
    qual = str(obj.__class__)
    try:
        qual = qual[qual.index("'") + 1:-2]
    except ValueError:
        pass
    qual = qual.split('.')
    if module in ['builtins', '__builtin__']:
        if qual[-1] == 'ellipsis':
            qual[-1] = 'EllipsisType'
        if _intypes(qual[-1]):
            module = 'types'
        qual = [module] + qual
    return qual