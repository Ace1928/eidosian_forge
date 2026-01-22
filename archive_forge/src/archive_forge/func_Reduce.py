from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
@_internal_types.wraps_class_method
def Reduce(self, env, instance, input, reduction_op):
    if input.ctype != self.T:
        raise TypeError(f'Invalid input type {input.ctype}. ({self.T} is expected.)')
    return _internal_types.Data(f'{instance.code}.Reduce({input.code}, {reduction_op.code})', input.ctype)