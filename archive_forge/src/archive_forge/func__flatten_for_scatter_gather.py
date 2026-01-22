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
def _flatten_for_scatter_gather(tensor_list, copy=False):
    """Flatten the tensor for gather/scatter operations.

    Args:
        tensor_list: the list of tensors to be scattered/gathered.
        copy: whether the copy the tensors in tensor_list into the buffer.

    Returns:
        The flattened tensor buffer.
    """
    if not tensor_list:
        raise RuntimeError('Received an empty list.')
    t = tensor_list[0]
    dtype = nccl_util.get_cupy_tensor_dtype(t)
    buffer_shape = [len(tensor_list)] + nccl_util.get_tensor_shape(t)
    device = nccl_util.get_tensor_device(t)
    with nccl_util.Device(device):
        buffer = cupy.empty(buffer_shape, dtype=dtype)
    if copy:
        for i, tensor in enumerate(tensor_list):
            nccl_util.copy_tensor(buffer[i], tensor)
    return buffer