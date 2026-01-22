import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _get_indices_and_dtypes(self, indices=None):
    if indices is None:
        indices = list(range(len(self._dtypes)))
    if not isinstance(indices, (tuple, list)):
        raise TypeError(f'Invalid indices type {type(indices)}')
    if len(indices) == 0:
        raise ValueError('Empty indices')
    if all((isinstance(i, str) for i in indices)):
        if self._names is None:
            raise ValueError(f'String indices provided {indices}, but this Staging Area was not created with names.')
        try:
            indices = [self._names.index(n) for n in indices]
        except ValueError:
            raise ValueError(f'Named index not in Staging Area names {self._names}')
    elif all((isinstance(i, int) for i in indices)):
        pass
    else:
        raise TypeError(f'Mixed types in indices {indices}. May only be str or int')
    dtypes = [self._dtypes[i] for i in indices]
    return (indices, dtypes)