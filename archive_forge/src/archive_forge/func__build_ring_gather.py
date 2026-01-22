import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _build_ring_gather(input_tensors, devices, num_subchunks, pred_by_s_d, rank_by_s_d, red_op):
    """Construct a subgraph for the first (reduction) pass of ring all-reduce.

  Args:
    input_tensors: a list of `tf.Tensor` 1D input tensors of same
      shape and type.
    devices: array of device name strings
    num_subchunks: number of subchunks each device should process in one tick.
    pred_by_s_d: as produced by _ring_permutations
    rank_by_s_d: as produced by _ring_permutations
    red_op: a binary operator for elementwise reduction

  Raises:
    ValueError: tensors must all be one dimensional.

  Returns:
    list of list of `tf.Tensor` of (partially) reduced values where
    exactly num_subchunks chunks at each device are fully reduced.
  """
    num_devices = len(input_tensors)
    if num_devices == 0:
        return []
    if num_devices == 1:
        return input_tensors
    shape = input_tensors[0].shape
    if 1 != len(shape):
        raise ValueError('input tensors must be 1D')
    num_chunks = num_devices * num_subchunks
    num_ticks = num_devices - 1
    chunks_by_dev = []
    split_pad_len = 0
    for d in range(0, num_devices):
        with ops.device(devices[d]):
            splits, split_pad_len = _padded_split(input_tensors[d], num_chunks)
            chunks_by_dev.append(splits)
    for tick in range(0, num_ticks):
        new_partial_reductions = [None for _ in range(0, num_chunks)]
        for d in range(0, num_devices):
            with ops.device(devices[d]):
                for s in range(0, num_subchunks):
                    rank = rank_by_s_d[s][d]
                    seg_index = (rank + num_devices - (2 + tick)) % num_devices
                    pred_dev = pred_by_s_d[s][d]
                    chunk_index = seg_index * num_subchunks + s
                    new_partial_reductions[chunk_index] = red_op(chunks_by_dev[pred_dev][chunk_index], chunks_by_dev[d][chunk_index])
        for d in range(0, num_devices):
            for s in range(0, num_subchunks):
                rank = rank_by_s_d[s][d]
                seg_index = (rank + num_devices - (2 + tick)) % num_devices
                chunk_index = seg_index * num_subchunks + s
                chunks_by_dev[d][chunk_index] = new_partial_reductions[chunk_index]
    return (chunks_by_dev, split_pad_len)