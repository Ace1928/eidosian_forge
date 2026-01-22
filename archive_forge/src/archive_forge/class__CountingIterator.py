from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
class _CountingIterator(_cuda_types.PointerBase):

    def __str__(self) -> str:
        value_type = self.child_type
        return f'thrust::counting_iterator<{value_type}>'