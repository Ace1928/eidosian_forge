import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _build_recursive_hd_gather(input_tensors, devices, red_op):
    """Construct the gather phase of recursive halving-doubling all-reduce.

  Args:
    input_tensors: list of `tf.Tensor` to be elementwise reduced.
    devices: a list of strings naming the devices hosting input_tensors,
      which will also be used to host the (partial) reduction values.
    red_op: a binary elementwise reduction Op.

  Returns:
    list of `tf.Tensor` which are the fully reduced tensor shards.

  Raises:
    ValueError: num_devices not a power of 2, or tensor len not divisible
    by 2 the proper number of times.
  """
    num_devices = len(devices)
    num_hops = int(math.log(num_devices, 2))
    if num_devices != 2 ** num_hops:
        raise ValueError('num_devices must be a power of 2')
    chunks = input_tensors
    for h in range(0, num_hops):
        span = 2 ** h
        group_size = span * 2
        new_chunks = [[] for _ in devices]
        for d in range(0, num_devices):
            if d % group_size >= group_size / 2:
                continue
            left_dev = devices[d]
            right_dev = devices[d + span]
            left_split = array_ops.split(chunks[d], 2)
            right_split = array_ops.split(chunks[d + span], 2)
            with ops.device(left_dev):
                new_chunks[d] = red_op(left_split[0], right_split[0])
            with ops.device(right_dev):
                new_chunks[d + span] = red_op(left_split[1], right_split[1])
        chunks = new_chunks
    return chunks