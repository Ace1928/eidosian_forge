import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _build_nccl_hybrid(input_tensors, red_op, upper_level_f):
    """Construct a subgraph for NCCL hybrid all-reduce.

  Args:
    input_tensors: list of `tf.Tensor` of same-shape and type values to
      be reduced.
    red_op: binary elementwise reduction operator.
    upper_level_f: function for reducing one value per worker, across
      workers.

  Returns:
    list of `tf.Tensor` of reduced values.

  Raises:
    ValueError: inputs not well-formed.
  """
    input_tensors, shape = _flatten_tensors(input_tensors)
    devices = [t.device for t in input_tensors]
    per_worker_devices, per_worker_values = _split_by_task(devices, input_tensors)
    num_workers = len(per_worker_devices)
    up_values = [None for w in range(0, num_workers)]
    up_devices = up_values[:]
    down_values = up_values[:]
    for w in range(0, num_workers):
        worker_values = build_nccl_all_reduce(per_worker_values[w], red_op)
        with ops.control_dependencies(worker_values):
            with ops.device(worker_values[0].device):
                up_values[w] = array_ops.identity(worker_values[0])
            up_devices[w] = per_worker_devices[w][0]
    level_2_output = upper_level_f(up_values)
    for w in range(0, num_workers):
        dst_tensors = []
        with ops.device(per_worker_devices[w][0]):
            broadcast_src = nccl_ops.broadcast(array_ops.identity(level_2_output[w]))
        for d in per_worker_devices[w]:
            with ops.device(d):
                dst_tensors.append(array_ops.identity(broadcast_src))
        down_values[w] = dst_tensors
    output_tensors = [v for sublist in down_values for v in sublist]
    if len(shape) != 1:
        output_tensors = _reshape_tensors(output_tensors, shape)
    return output_tensors