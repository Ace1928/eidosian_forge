from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
Cast a list of losses to a common dtype.

  If any loss is floating-point, they will all be casted to the most-precise
  floating-point loss. Otherwise the losses are not casted. We also skip casting
  losses if there are any complex losses.

  Args:
    losses: A list of losses.

  Returns:
    `losses`, but they have been casted to a common dtype.
  