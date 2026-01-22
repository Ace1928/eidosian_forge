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
def _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists):
    """Check the compatibility between tensor input and tensor list input."""
    if not tensors or not isinstance(tensors, list):
        raise RuntimeError("The first argument 'tensors' expects a list of tensors.")
    if not tensor_lists or not isinstance(tensor_lists, list):
        raise RuntimeError("The second argument 'tensor_lists' expects a list of tensor list.")
    dtype = nccl_util.get_nccl_tensor_dtype(tensors[0])
    shape = nccl_util.get_tensor_shape(tensors[0])
    for i, tensor_list in enumerate(tensor_lists):
        dt = nccl_util.get_nccl_tensor_dtype(tensors[i])
        if dt != dtype:
            raise RuntimeError("All tensor operands to scatter/gather must have the same dtype. Got '{}' and '{}'.".format(dt, dtype))
        s = nccl_util.get_tensor_shape(tensors[i])
        if s != shape:
            raise RuntimeError("All tensor operands to scatter/gather must have the same shape. Got '{}' and '{}'.".format(s, shape))
        for t in tensor_lists[i]:
            dt = nccl_util.get_nccl_tensor_dtype(t)
            if dt != dtype:
                raise RuntimeError("All tensor operands to scatter/gather must have the same dtype. Got '{}' and '{}'.".format(dt, dtype))
            s = nccl_util.get_tensor_shape(t)
            if s != shape:
                raise RuntimeError("All tensor operands to scatter/gather must have the same shape. Got '{}' and '{}'.".format(s, shape))