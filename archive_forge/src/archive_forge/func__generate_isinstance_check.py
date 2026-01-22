import typing
from typing import Protocol
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _generate_isinstance_check(expected_types):

    def inner(values):
        for v in nest.flatten(values):
            if not (isinstance(v, expected_types) or (isinstance(v, np.ndarray) and issubclass(v.dtype.type, expected_types))):
                _check_failed(v)
    return inner