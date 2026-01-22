import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _build_ring_scatter(pred_by_s_d, rank_by_s_d, chunks_by_dev):
    """Construct subgraph for second (scatter) pass of ring all-reduce.

  Args:
    pred_by_s_d: as produced by _ring_permutations
    rank_by_s_d: as produced by _ring_permutations
    chunks_by_dev: list of list of `tf.Tensor` indexed by ints
      (device, chunk)

  Raises:
    ValueError: chunks_by_dev is not well-formed

  Returns:
    list of `tf.Tensor` which are the fully reduced tensors, one
    at each device corresponding to the outer dimension of chunks_by_dev.
  """
    num_devices = len(chunks_by_dev)
    num_chunks = len(chunks_by_dev[0])
    if 0 != num_chunks % num_devices:
        raise ValueError('Expect number of chunks per device to be divisible by num_devices')
    num_subchunks = int(num_chunks / num_devices)
    num_ticks = num_devices - 1
    for tick in range(0, num_ticks):
        passed_values = [None for _ in range(0, num_chunks)]
        for d in range(0, num_devices):
            with ops.colocate_with(chunks_by_dev[d][0]):
                for s in range(0, num_subchunks):
                    rank = rank_by_s_d[s][d]
                    seg_index = (rank + num_devices - (1 + tick)) % num_devices
                    pred_dev = pred_by_s_d[s][d]
                    chunk_index = seg_index * num_subchunks + s
                    passed_values[chunk_index] = array_ops.identity(chunks_by_dev[pred_dev][chunk_index])
        for d in range(0, num_devices):
            for s in range(0, num_subchunks):
                rank = rank_by_s_d[s][d]
                seg_index = (rank + num_devices - (1 + tick)) % num_devices
                chunk_index = seg_index * num_subchunks + s
                chunks_by_dev[d][chunk_index] = passed_values[chunk_index]
    output = []
    for x in chunks_by_dev:
        with ops.colocate_with(x[0]):
            output.append(array_ops.concat(x, 0))
    return output