from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
class _WarpReduceType(_CubReduceBaseType):

    def __init__(self, T) -> None:
        self.T = _cuda_typerules.to_ctype(T)
        self.TempStorage = _TempStorageType(self)
        super().__init__()

    def __str__(self) -> str:
        namespace = _get_cub_namespace()
        return f'{namespace}::WarpReduce<{self.T}>'