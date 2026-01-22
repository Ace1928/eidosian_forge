import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
Get the gradient tensors that this object is aware of.

    Returns:
      A dict mapping x-tensor names to gradient tensor objects. x-tensor refers
      to the tensors on the denominator of the differentation.
    