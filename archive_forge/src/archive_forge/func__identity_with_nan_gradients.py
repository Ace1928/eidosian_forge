from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
@custom_gradient.custom_gradient
def _identity_with_nan_gradients(x):
    """Function whose gradient is NaN iff `have_nan_gradients` is True."""
    x = array_ops.identity(x)

    def grad(dx):
        return cond.cond(have_nan_gradients, lambda: dx * float('NaN'), lambda: dx)
    return (x, grad)