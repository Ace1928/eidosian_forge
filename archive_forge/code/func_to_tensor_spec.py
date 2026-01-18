from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def to_tensor_spec(input_spec, default_dtype=None):
    """Converts a Keras InputSpec object to a TensorSpec."""
    default_dtype = default_dtype or backend.floatx()
    if isinstance(input_spec, InputSpec):
        dtype = input_spec.dtype or default_dtype
        return tensor_spec.TensorSpec(to_tensor_shape(input_spec), dtype)
    return tensor_spec.TensorSpec(None, default_dtype)