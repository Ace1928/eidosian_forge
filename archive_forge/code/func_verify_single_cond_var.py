import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def verify_single_cond_var(name, body_var, orelse_var):
    """Verifies whether body_var and orelse_var are consistent."""
    if body_var is None:
        raise ValueError("'{}' is None at the end of the main branch.".format(name))
    if orelse_var is None:
        raise ValueError("'{}' is None at the end of the else branch.".format(name))
    if isinstance(body_var, (bool, int, float, str, np.ndarray)):
        body_var = tensor_conversion.convert_to_tensor_v2(body_var)
    if isinstance(orelse_var, (bool, int, float, str, np.ndarray)):
        orelse_var = tensor_conversion.convert_to_tensor_v2(orelse_var)
    if not tensor_util.is_tf_type(body_var) or not tensor_util.is_tf_type(orelse_var):
        return
    if not hasattr(body_var, 'dtype') or not hasattr(orelse_var, 'dtype'):
        return
    if body_var.dtype != orelse_var.dtype:
        raise TypeError("'{}' has dtype {} in the main branch, but dtype {} in the else branch".format(name, body_var.dtype.name, orelse_var.dtype.name))