from llvmlite.ir.transforms import CallVisitor
from numba.core import types
def _legalize(module, dmm, fndesc):
    """
    Legalize the code in the module.
    Returns True if the module is legal for the rewrite pass that removes
    unnecessary refcounts.
    """

    def valid_output(ty):
        """
        Valid output are any type that does not need refcount
        """
        model = dmm[ty]
        return not model.contains_nrt_meminfo()

    def valid_input(ty):
        """
        Valid input are any type that does not need refcount except Array.
        """
        return valid_output(ty) or isinstance(ty, types.Array)
    try:
        nmd = module.get_named_metadata('numba_args_may_always_need_nrt')
    except KeyError:
        pass
    else:
        if len(nmd.operands) > 0:
            return False
    argtypes = fndesc.argtypes
    restype = fndesc.restype
    calltypes = fndesc.calltypes
    for argty in argtypes:
        if not valid_input(argty):
            return False
    if not valid_output(restype):
        return False
    for callty in calltypes.values():
        if callty is not None and (not valid_output(callty.return_type)):
            return False
    for fn in module.functions:
        if fn.name.startswith('NRT_'):
            if fn.name not in _accepted_nrtfns:
                return False
    return True