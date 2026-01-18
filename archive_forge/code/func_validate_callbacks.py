import functools
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def validate_callbacks(input_callbacks, optimizer):
    """Validate whether given callbacks are supported by DistributionStrategy.

  Args:
    input_callbacks: List of callbacks passed by the user to fit.
    optimizer: Optimizer instance used to train the model.

  Raises:
    ValueError: If `LearningRateScheduler` or `ReduceLROnPlateau` is one of the
        callbacks passed.
    ValueError: If `write_grads` is one of the parameters passed as part of the
        TensorBoard callback.
  """
    if input_callbacks:
        for callback in input_callbacks:
            if isinstance(callback, (callbacks.LearningRateScheduler, callbacks.ReduceLROnPlateau)):
                if not isinstance(optimizer, optimizer_v2.OptimizerV2):
                    raise ValueError('You must specify a Keras Optimizer V2 when using %s callback with DistributionStrategy.' % callback)
            if isinstance(callback, callbacks.TensorBoard):
                if getattr(callback, 'write_grads', False):
                    logging.warning(UserWarning('`write_grads` in the TensorBoard callback is not supported when using DistributionStrategy. Setting `write_grads` to `False`.'))
                    callback.write_grads = False