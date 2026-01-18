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
def tpu_core_ids_to_locations(self, tpu_core_ids):
    """Translates TPU core IDs to TPU core locations.

    Args:
      tpu_core_ids: A list of TPU core IDs. Each one is an unsigned integer.

    Returns:
      A list of corresponding TPU core locations.
    """
    return _pywrap_dtensor_device.TPUCoreIDsToLocations(context.context()._handle, self._device_info, tpu_core_ids)