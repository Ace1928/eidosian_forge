import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
def uses_keras_history(tensors):
    """Check if at least one Tensor originates from a `keras.Input`.

  This is `True` if at least one Tensor has its origin in a `keras.Input`.
  Any Tensor that originates from a `keras.Input` will have a dependency
  Tensor with a `_keras_history` attribute attached. Tensors that have
  already been checked to not originate from a `keras.Input`
  are marked as `_keras_history_checked`.

  Args:
    tensors: An arbitrary nested structure of Tensors.

  Returns:
    Bool, whether at least one Tensor originates from a `keras.Input`.
  """
    checked_tensors = set()
    tensors_to_check = nest.flatten(tensors)
    while tensors_to_check:
        new_tensors_to_check = []
        for tensor in tensors_to_check:
            if id(tensor) in checked_tensors:
                continue
            checked_tensors.add(id(tensor))
            if getattr(tensor, '_keras_history_checked', None) is not None:
                continue
            if getattr(tensor, '_keras_history', None) is not None:
                return True
            try:
                new_tensors_to_check.extend(tensor.op.inputs)
            except AttributeError:
                pass
        tensors_to_check = new_tensors_to_check
    mark_checked(tensors)
    return False