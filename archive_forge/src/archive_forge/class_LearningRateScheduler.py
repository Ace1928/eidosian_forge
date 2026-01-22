import collections
import copy
import csv
import json
import os
import re
import sys
import time
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options as checkpoint_options_lib
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class LearningRateScheduler(Callback):
    """Learning rate scheduler.

  At the beginning of every epoch, this callback gets the updated learning rate
  value from `schedule` function provided at `__init__`, with the current epoch
  and current learning rate, and applies the updated learning rate
  on the optimizer.

  Args:
    schedule: a function that takes an epoch index (integer, indexed from 0)
        and current learning rate (float) as inputs and returns a new
        learning rate as output (float).
    verbose: int. 0: quiet, 1: update messages.

  Example:

  >>> # This function keeps the initial learning rate for the first ten epochs
  >>> # and decreases it exponentially after that.
  >>> def scheduler(epoch, lr):
  ...   if epoch < 10:
  ...     return lr
  ...   else:
  ...     return lr * tf.math.exp(-0.1)
  >>>
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> round(model.optimizer.lr.numpy(), 5)
  0.01

  >>> callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=15, callbacks=[callback], verbose=0)
  >>> round(model.optimizer.lr.numpy(), 5)
  0.00607

  """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:
            lr = float(backend.get_value(self.model.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:
            lr = self.schedule(epoch)
        if not isinstance(lr, (tensor_lib.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')
        if isinstance(lr, tensor_lib.Tensor) and (not lr.dtype.is_floating):
            raise ValueError('The dtype of Tensor should be float')
        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)