import contextlib
import torch
from torch.distributed import ReduceOp
class NCCLBackendWithBFloat16(NCCLBackend):

    def _get_nccl_dtype_and_count(self, array, count=None):
        nccl_dtype, count = super()._get_nccl_dtype_and_count(array, count)
        torch_dtype = getattr(array, '_torch_dtype', None)
        if torch_dtype is torch.bfloat16:
            nccl_dtype = nccl.NCCL_BFLOAT16
        return (nccl_dtype, count)

    def barrier(self) -> None:
        raise RuntimeError('Currently, CuPy NCCL barrier is not supported since the TCP store is immediately stopped after the initialization.')