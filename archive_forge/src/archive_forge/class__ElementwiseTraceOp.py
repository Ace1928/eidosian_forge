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
class _ElementwiseTraceOp:
    """Ufunc or elementwise kernel with types.
    """

    def __init__(self, ufunc_routines, in_params, out_params, ashape):
        _fusion_thread_local.check_not_runtime()
        assert isinstance(ufunc_routines, list)
        assert all((isinstance(r, _UfuncRoutine) for r in ufunc_routines))
        assert isinstance(ashape, tuple)
        self.ops = ufunc_routines
        self.in_params = _VariableSet(*in_params)
        self.out_params = _VariableSet(*out_params)
        self.ashape = ashape

    @property
    def params(self):
        """Returns the set of all variable the loop uses.
        """
        res = _VariableSet()
        for op in self.ops:
            res += _VariableSet(*op.in_params)
            res += _VariableSet(*op.out_params)
        return res

    @staticmethod
    def _emit_declaration(params, in_params):
        """Returns a tuple of size 2.

        1. CUDA code: declaring local variables.
            2. The set of arrays which require indexer.
        """
        _fusion_thread_local.check_not_runtime()
        indexed_arrays = _VariableSet()
        code = []
        for var in params:
            if var in in_params:
                if isinstance(var, _TraceArray):
                    indexed_arrays.add(var)
                    f = '${type} ${lvar} = ${var}[${indexer}.get()];'
                else:
                    f = '${type} ${lvar} = ${var};'
            else:
                f = '${type} ${lvar};'
            code.append(var.format(f))
        return (code, indexed_arrays)

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

    @staticmethod
    def _emit_set_index(indexed_params, tid):
        """Returns a CUDA code: setting a raw index to indexers.
        """
        _fusion_thread_local.check_not_runtime()
        assert isinstance(indexed_params, _VariableSet)
        return [p.format('${indexer}.set(${tid});', tid=tid) for p in indexed_params]

    def emit_code(self):
        _fusion_thread_local.check_not_runtime()
        declaration, s1 = self._emit_declaration(self.params, self.in_params)
        operation = [op.emit_call_code() for op in self.ops]
        after_operation, s2 = self._emit_after_operation(self.out_params)
        index_name = 'i'
        indexed_array = s1 + s2
        indexer_name = next(iter(indexed_array)).indexer_name
        indexer_setup = self._emit_set_index(indexed_array, index_name)
        return _codeblock.CodeBlock('CUPY_FOR({}, {}.size())'.format(index_name, indexer_name), indexer_setup + declaration + operation + after_operation)

    def emit_preamble_codes(self):
        return [subm.preamble for subm in self.ops if subm.preamble != '']

    def emit_submodule_codes(self):
        return [str(subm.emit_code()) for subm in self.ops]