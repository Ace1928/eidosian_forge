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
def _tensor_conversion(var, dtype=None, name=None, as_ref=False):
    if as_ref:
        raise ValueError('You may be using variable created under distribute strategy in TF 1.x control flows. Try explicitly converting the variable to Tensor using variable.read_value(), or switch to TF 2.x.')
    return ops.convert_to_tensor(var.read_value(), dtype=dtype, name=name, as_ref=as_ref)