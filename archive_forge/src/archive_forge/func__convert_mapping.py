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
def _convert_mapping(value, expected_type, path, context):
    """Converts `value` to a mapping with type `expected_type`."""
    if not isinstance(value, typing.Mapping):
        raise TypeError(f'{''.join(path)}: expected mapping, got {type(value).__name__!r}')
    key_type, value_type = type_annotations.get_generic_type_args(expected_type)
    return immutable_dict.ImmutableDict([(_convert_value(k, key_type, path + ('[<key>]',), context), _convert_value(v, value_type, path + (f'[{k!r}]',), context)) for k, v in value.items()])