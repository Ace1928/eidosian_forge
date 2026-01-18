import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def varnames(func):
    """get names of variables defined by func

    returns a tuple (local vars, local vars referrenced by nested functions)"""
    func = code(func)
    if not iscode(func):
        return ()
    return (func.co_varnames, func.co_cellvars)