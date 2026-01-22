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
def _report_field_mismatches(fields, field_values):
    """Raises an exception with mismatches between fields and field_values."""
    expected = set((f.name for f in fields))
    actual = set(field_values)
    extra = actual - expected
    if extra:
        raise ValueError(f'Got unexpected fields: {extra}')
    missing = expected - actual
    if missing:
        raise ValueError(f'Missing required fields: {missing}')