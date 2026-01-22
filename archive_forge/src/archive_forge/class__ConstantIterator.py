from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
class _ConstantIterator(_cuda_types.PointerBase):

    def __str__(self) -> str:
        value_type = self.child_type
        return f'thrust::constant_iterator<{value_type}>'