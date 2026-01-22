import contextlib
import logging
import threading
from typing import Any, List, Sequence, Set
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import _pywrap_utils
def _get_stats(self):
    """Returns the number of cache hit and miss for function compilation.

    Returns:
      A dictionary.
        'miss': number of cache misses;
        'hit': number of cache hits; and
        'size': size of cache;
      miss count.
    """
    return _pywrap_dtensor_device.GetStats(context.context()._handle, self._device_info)