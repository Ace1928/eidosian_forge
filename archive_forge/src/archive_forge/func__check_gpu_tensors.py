import logging
import datetime
import time
import ray
import cupy
from ray.util.collective.const import ENV
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
from ray.util.collective.collective_group.cuda_stream import get_stream_pool
def _check_gpu_tensors(tensors):
    """Check all tensors are distributed on different GPUs."""
    if not tensors or not isinstance(tensors, list):
        raise RuntimeError("'tensors' must be a nonempty list.")
    if len(tensors) > nccl_util.get_num_gpus():
        raise RuntimeError('Tensor list cannot be larger than the numberof available GPUs. Got {} > {}.'.format(len(tensors), nccl_util.get_num_gpus()))
    t0 = tensors[0]
    dt = nccl_util.get_nccl_tensor_dtype(t0)
    s = nccl_util.get_tensor_shape(t0)
    d = nccl_util.get_tensor_device(t0)
    for i, t in enumerate(tensors):
        if i == 0:
            continue
        dtype = nccl_util.get_nccl_tensor_dtype(t)
        if dt != dtype:
            raise RuntimeError("Tensors must have identical dtype. Got: '{}'.".format(dtype))
        shape = nccl_util.get_tensor_shape(t)
        if s != shape:
            raise RuntimeError("Tensor must have identical shape. Got: '{}'.".format(shape))
        device = nccl_util.get_tensor_device(t)
        if device == d:
            raise RuntimeError('Tensor must be on distinct GPUs.')