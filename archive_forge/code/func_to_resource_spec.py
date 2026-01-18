from typing import List, Optional
from tensorflow.core.function import trace_type
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
def to_resource_spec(traced_input):
    try:
        handle_data = traced_input.dtype._handle_data.shape_inference
        shape_and_type = handle_data.shape_and_type[0]
        spec = tensor_spec.TensorSpec(shape=shape_and_type.shape, dtype=shape_and_type.dtype)
        return spec
    except Exception as e:
        raise ValueError('Fail to convert tf.Variable list to TensorSpec list. The error is: %s' % e) from e