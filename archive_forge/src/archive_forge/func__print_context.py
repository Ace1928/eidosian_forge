from typing import List, Optional, Tuple
from absl import logging
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def _print_context(num_global_devices: int, num_clients: int, client_id: int, device_type: str, mesh: layout.Mesh) -> None:
    logging.info('This is client %d of %d clients', client_id, num_clients)
    logging.info('Number of global %s devices: %d', device_type.upper(), num_global_devices)
    logging.info('Global device IDs: %s', mesh.global_device_ids())
    logging.info('Local device IDs: %s', mesh.local_device_ids())
    logging.info('Local devices: %s', mesh.local_devices())