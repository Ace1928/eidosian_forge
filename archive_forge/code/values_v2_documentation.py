import copy
import weakref
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
Represents variables that are replicated.

  It behaves exactly as a normal variable, but uses corresponding variable
  handle based on the context.
  - In each replica, it uses the handle from that replica.
  - In tpu.replicate(), it uses the replicated handle.
  - Otherwise, it uses the handle from the primary replica.

  Note that it doesn't synchronize automatically as the old DistributedVariable
  in values.py.
  