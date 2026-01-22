import collections
import collections.abc
import enum
import typing
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.util import type_annotations
def _convert_value(value, expected_type, path, context=_ConversionContext.VALUE):
    """Type-checks and converts a value.

  Args:
    value: The value to type-check.
    expected_type: The expected type for the value.
    path: Tuple of `str` naming the value (used for exception messages).
    context: _ConversionContext, indicates what kind of value we are converting.

  Returns:
    A copy of `value`, converted to the expected type.

  Raises:
    TypeError: If `value` can not be converted to the expected type.
  """
    assert isinstance(path, tuple)
    if expected_type is None:
        expected_type = _NoneType
    if expected_type is tensor.Tensor:
        return _convert_tensor(value, path, context)
    elif isinstance(expected_type, type) and _issubclass(expected_type, composite_tensor.CompositeTensor):
        return _convert_composite_tensor(value, expected_type, path, context)
    elif expected_type is tensor_shape.TensorShape:
        try:
            return tensor_shape.as_shape(value)
        except TypeError as e:
            raise TypeError(f"{''.join(path)}: expected 'tf.TensorShape', got {type(value).__name__!r}") from e
    elif expected_type is dtypes.DType:
        try:
            return dtypes.as_dtype(value)
        except TypeError as e:
            raise TypeError(f"{''.join(path)}: expected 'tf.DType', got {type(value).__name__!r}") from e
    elif expected_type in (int, float, bool, str, bytes, _NoneType):
        if not isinstance(value, expected_type):
            raise TypeError(f'{''.join(path)}: expected {expected_type.__name__!r}, got {type(value).__name__!r}')
        return value
    elif type_annotations.is_generic_tuple(expected_type):
        return _convert_tuple(value, expected_type, path, context)
    elif type_annotations.is_generic_mapping(expected_type):
        return _convert_mapping(value, expected_type, path, context)
    elif type_annotations.is_generic_union(expected_type):
        return _convert_union(value, expected_type, path, context)
    else:
        raise TypeError(f'{''.join(path)}: Unsupported type annotation {expected_type!r}')