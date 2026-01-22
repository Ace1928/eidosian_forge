import six
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _as_tuple(value):
    if not nest.is_nested(value):
        return value
    return tuple([_as_tuple(v) for v in value])