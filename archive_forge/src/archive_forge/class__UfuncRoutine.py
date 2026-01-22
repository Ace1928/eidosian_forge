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
class _UfuncRoutine:
    """A device function for single elementwise operations.
    """

    def __init__(self, name, ufunc, routine_code, in_params, out_params, compute_dtypes):
        assert isinstance(name, str)
        assert isinstance(ufunc, _kernel.ufunc)
        assert isinstance(routine_code, str)
        assert isinstance(compute_dtypes, tuple)
        assert all((isinstance(t, numpy.dtype) for t in compute_dtypes))
        assert isinstance(in_params, list)
        assert all((isinstance(p, _TraceVariable) for p in in_params))
        assert isinstance(out_params, list)
        assert all((isinstance(p, _TraceArray) for p in out_params))
        self.name = name
        self.in_params = in_params
        self.out_params = out_params
        self.preamble = ufunc._preamble
        self.routine_code = routine_code
        self.compute_dtypes = compute_dtypes

    def emit_code(self):
        """Returns a CUDA device function code.

        Returns a string like:
        ```
        __device__ void cupy_add_0(int &in0_, float &in1_, double &out0_) {
            typedef double in0_type;
            typedef double in1_type;
            typedef double out0_type;
            double in0 = (double) in0_;
            double in1 = (double) in1_;
            double out0 = (double) out0_;
            out0 = in0 + in1;
            out0_ = out0;
        }
        ```
        """
        nin = len(self.in_params)
        dtypes = self.compute_dtypes
        assert len(self.in_params) == len(self.compute_dtypes[:nin])
        in_params = [(get_typename(p.dtype), get_typename(t), 'in{}'.format(i)) for i, (p, t) in enumerate(zip(self.in_params, dtypes[:nin]))]
        out_params = [(get_typename(p.dtype), get_typename(t), 'out{}'.format(i)) for i, (p, t) in enumerate(zip(self.out_params, dtypes[nin:]))]
        params = in_params + out_params
        params_code = ', '.join(['{} &{}_'.format(t, s) for t, _, s in params])
        typedef = ['typedef {} {}_type;'.format(t, s) for _, t, s in params]
        read = ['{} {} = ({}) {}_;'.format(t, s, t, s) for _, t, s in params]
        write = ['{}_ = {};'.format(s, s) for _, _, s in out_params]
        return _codeblock.CodeBlock('__device__ void {}({})'.format(self.name, params_code), typedef + read + [self.routine_code + ';'] + write)

    def emit_call_code(self):
        params = self.in_params + self.out_params
        return '{op_name}({params});'.format(op_name=self.name, params=', '.join([var.lvar_name for var in params]))