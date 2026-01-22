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
def _emit_set_index(indexed_params, tid):
    """Returns a CUDA code: setting a raw index to indexers.
        """
    _fusion_thread_local.check_not_runtime()
    assert isinstance(indexed_params, _VariableSet)
    return [p.format('${indexer}.set(${tid});', tid=tid) for p in indexed_params]