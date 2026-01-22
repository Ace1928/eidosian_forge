import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.types import core
def _should_cast(self):
    """Returns True if this variable should be casted when accessed."""
    autocast_dtype = getattr(_autocast_dtype, 'dtype', None)
    return autocast_dtype is not None and self.dtype != autocast_dtype