import string
import numpy
from cupy._core import _codeblock
from cupy._core._fusion_variable import _TraceVariable
from cupy._core._fusion_variable import _TraceArray
from cupy._core._fusion_variable import _VariableSet
from cupy._core import _fusion_thread_local
from cupy._core import _kernel
from cupy._core import _reduction
from cupy._core._scalar import get_typename
@staticmethod
def _emit_after_operation(out_params):
    """Returns a tuple of size 2.
        1. CUDA code: writing the results of operations back to global memory.
        2. The set of arrays which require indexer.
        """
    _fusion_thread_local.check_not_runtime()
    indexed_arrays = _VariableSet()
    codes = []
    for var in out_params:
        if isinstance(var, _TraceArray):
            indexed_arrays.add(var)
            f = '${var}[${indexer}.get()] = ${lvar};'
        else:
            f = '${var} = ${lvar};'
        codes.append(var.format(f))
    return (codes, indexed_arrays)