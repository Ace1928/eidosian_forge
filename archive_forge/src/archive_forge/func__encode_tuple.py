import collections
import enum
import json
import numpy as np
import wrapt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
def _encode_tuple(x):
    if isinstance(x, tuple):
        return {'class_name': '__tuple__', 'items': tuple((_encode_tuple(i) for i in x))}
    elif isinstance(x, list):
        return [_encode_tuple(i) for i in x]
    elif isinstance(x, dict):
        return {key: _encode_tuple(value) for key, value in x.items()}
    else:
        return x