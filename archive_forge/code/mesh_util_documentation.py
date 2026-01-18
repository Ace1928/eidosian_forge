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
Runs a barrier on the mesh.

  Upon returning from the barrier, all operations run before the barrier
  would have completed across all clients. Currently we allocate a fully
  sharded tensor with mesh shape and run an all_reduce on it.

  Example:

  A barrier can be used before application exit to ensure completion of pending
  ops.

  ```python

  x = [1, 2, 3]
  x = dtensor.relayout(x, dtensor.Layout.batch_sharded(mesh, 'batch', 1))
  dtensor.barrier(mesh)

  # At this point all devices on all clients in the mesh have completed
  # operations before the barrier. Therefore it is OK to tear down the clients.
  sys.exit()
  ```

  Args:
    mesh: The mesh to run the barrier on.
    barrier_name: The name of the barrier. Mainly used for logging purpose.
    timeout_in_ms: The timeout of the barrier in ms. If omitted, blocks
      indefinitely till the barrier is reached from all clients.
  