from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data
def _assert_pointer_type(a: _Data) -> None:
    if not isinstance(a.ctype, _cuda_types.PointerBase):
        raise TypeError(f'`{a.code}` must be of pointer type: `{a.ctype}`')