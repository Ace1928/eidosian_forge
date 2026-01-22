import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
def _padding_values_or_default(padding_values, input_dataset):
    """Returns padding values with None elements replaced with default values."""

    def make_zero(t):
        if t.base_dtype == dtypes.string:
            return ''
        elif t.base_dtype == dtypes.variant:
            raise TypeError("Unable to create default padding value for a component of type 'variant'.")
        elif t.base_dtype == dtypes.bfloat16:
            return constant_op.constant(0, dtype=dtypes.bfloat16)
        else:
            return np.zeros_like(t.as_numpy_dtype())

    def value_or_default(value, default):
        return default if value is None else value
    default_padding = nest.map_structure(make_zero, dataset_ops.get_legacy_output_types(input_dataset))
    return nest.map_structure_up_to(padding_values, value_or_default, padding_values, default_padding)