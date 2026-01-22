from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
class _CubFunctor(_internal_types.BuiltinFunc):

    def __init__(self, name):
        namespace = _get_cub_namespace()
        self.fname = f'{namespace}::{name}()'

    def call_const(self, env):
        return _internal_types.Data(self.fname, _cuda_types.Unknown(label='cub_functor'))