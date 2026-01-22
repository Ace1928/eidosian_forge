import functools
import time
from typing import List, Optional, Dict
import numpy as np
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.util.tf_export import tf_export
def _create_device_array(shape, device_type, host_id, local_device_ids=None):
    """Returns ID and device lists that can be used to create a mesh."""
    num_global_devices = config.num_global_devices(device_type)
    global_device_ids = np.arange(num_global_devices).reshape(shape)
    local_device_list = config.local_devices(device_type)
    num_local_devices = len(local_device_list)
    local_device_ids = [x + host_id * num_local_devices for x in range(num_local_devices)] if not local_device_ids else local_device_ids
    return (global_device_ids, local_device_ids, local_device_list)