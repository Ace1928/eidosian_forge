from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
class _ThreadGroup(_cuda_types.TypeBase):
    """ Base class for all cooperative groups. """
    child_type = None

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.child_type}'

    def _sync(self, env, instance):
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.sync()', _cuda_types.void)