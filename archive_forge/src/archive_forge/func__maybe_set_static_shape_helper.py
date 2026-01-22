import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_random_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _maybe_set_static_shape_helper(tensor, shape, postfix_tensor):
    if not context.executing_eagerly() and ops.get_default_graph().building_function and (not tensor.shape.is_fully_defined()):
        shape = shape_util.shape_tensor(shape)
        const_shape = tensor_util.constant_value_as_shape(shape)
        postfix_tensor = ops.convert_to_tensor(postfix_tensor)
        tensor.set_shape(const_shape.concatenate(postfix_tensor.shape))