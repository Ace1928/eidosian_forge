from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
class _BlockReduceType(_CubReduceBaseType):

    def __init__(self, T, BLOCK_DIM_X: int) -> None:
        self.T = _cuda_typerules.to_ctype(T)
        self.BLOCK_DIM_X = BLOCK_DIM_X
        self.TempStorage = _TempStorageType(self)
        super().__init__()

    def __str__(self) -> str:
        namespace = _get_cub_namespace()
        return f'{namespace}::BlockReduce<{self.T}, {self.BLOCK_DIM_X}>'