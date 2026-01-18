from llvmlite.ir.transforms import CallVisitor
from numba.core import types
def valid_output(ty):
    """
        Valid output are any type that does not need refcount
        """
    model = dmm[ty]
    return not model.contains_nrt_meminfo()