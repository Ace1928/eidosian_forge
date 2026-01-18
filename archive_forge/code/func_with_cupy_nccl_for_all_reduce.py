import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
@contextlib.contextmanager
def with_cupy_nccl_for_all_reduce():
    """use CuPy nccl instead of torch.distributed for all reduce"""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        yield
    else:
        global _ENABLE_CUPY_FOR_ALL_REDUCE
        old = _ENABLE_CUPY_FOR_ALL_REDUCE
        _ENABLE_CUPY_FOR_ALL_REDUCE = True
        stream = torch.cuda.current_stream()
        with cupy_utils.set_cupy_stream(stream):
            yield
        _ENABLE_CUPY_FOR_ALL_REDUCE = old