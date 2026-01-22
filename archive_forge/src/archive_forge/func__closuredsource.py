import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _closuredsource(func, alias=''):
    """get source code for closured objects; return a dict of 'name'
    and 'code blocks'"""
    from .detect import freevars
    free_vars = freevars(func)
    func_vars = {}
    for name, obj in list(free_vars.items()):
        if not isfunction(obj):
            free_vars[name] = getsource(obj, force=True, alias=name)
            continue
        fobj = free_vars.pop(name)
        src = getsource(fobj, alias)
        if not src.lstrip().startswith('@'):
            src = importable(fobj, alias=name)
            org = getsource(func, alias, enclosing=False, lstrip=True)
            src = (src, org)
        else:
            org = getsource(func, enclosing=True, lstrip=False)
            src = importable(fobj, alias, source=True)
            src = (org, src)
        func_vars[name] = src
    src = ''.join(free_vars.values())
    if not func_vars:
        org = getsource(func, alias, force=True, enclosing=False, lstrip=True)
        src = (src, org)
    else:
        src = (src, None)
    func_vars[None] = src
    return func_vars