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
def tpu_core_locations_to_ids(self, tpu_core_locations):
    """Translates TPU core locations to TPU core IDs.

    Args:
      tpu_core_locations: A list of TPU core locations. Each one is a list of
        four unsigned integers, [x, y, z, core].

    Returns:
      A list of corresponding TPU core IDs.
    """
    return _pywrap_dtensor_device.TPUCoreLocationsToIDs(context.context()._handle, self._device_info, tpu_core_locations)