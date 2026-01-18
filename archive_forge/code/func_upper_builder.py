import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def upper_builder(tensors):
    return build_shuffle_all_reduce(tensors, second_gather_devices, red_op, un_op)