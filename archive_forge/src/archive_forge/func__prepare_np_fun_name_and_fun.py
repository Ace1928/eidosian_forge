import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _prepare_np_fun_name_and_fun(np_fun_name, np_fun):
    """Mutually propagates information between `np_fun_name` and `np_fun`.

  If one is None and the other is not, we'll try to make the former not None in
  a best effort.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: the numpy function whose docstring will be used.

  Returns:
    Processed `np_fun_name` and `np_fun`.
  """
    if np_fun_name is not None:
        assert isinstance(np_fun_name, str)
    if np_fun is not None:
        assert not isinstance(np_fun, str)
    if np_fun is None:
        assert np_fun_name is not None
        try:
            np_fun = getattr(np, str(np_fun_name))
        except AttributeError:
            np_fun = None
    if np_fun_name is None:
        assert np_fun is not None
        np_fun_name = np_fun.__name__
    return (np_fun_name, np_fun)