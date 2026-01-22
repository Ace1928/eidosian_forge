import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
@tf_export('config.LogicalDeviceConfiguration', 'config.experimental.VirtualDeviceConfiguration')
class LogicalDeviceConfiguration(collections.namedtuple('LogicalDeviceConfiguration', ['memory_limit', 'experimental_priority', 'experimental_device_ordinal'])):
    """Configuration class for a logical devices.

  The class specifies the parameters to configure a `tf.config.PhysicalDevice`
  as it is initialized to a `tf.config.LogicalDevice` during runtime
  initialization. Not all fields are valid for all device types.

  See `tf.config.get_logical_device_configuration` and
  `tf.config.set_logical_device_configuration` for usage examples.

  Fields:
    memory_limit: (optional) Maximum memory (in MB) to allocate on the virtual
      device. Currently only supported for GPUs.
    experimental_priority: (optional) Priority to assign to a virtual device.
      Lower values have higher priorities and 0 is the default.
      Within a physical GPU, the GPU scheduler will prioritize ops on virtual
      devices with higher priority. Currently only supported for Nvidia GPUs.
    experimental_device_ordinal: (optional) Ordinal number to order the virtual
    device.
      LogicalDevice with lower ordinal number will receive a lower device id.
      Physical device id and location in the list is used to break ties.
      Currently only supported for Nvidia GPUs.
  """

    def __new__(cls, memory_limit=None, experimental_priority=None, experimental_device_ordinal=None):
        return super().__new__(cls, memory_limit, experimental_priority, experimental_device_ordinal)