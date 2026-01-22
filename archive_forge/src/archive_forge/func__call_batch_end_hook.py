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
def _call_batch_end_hook(self, mode, batch, logs):
    """Helper function for `on_*_batch_end` methods."""
    hook_name = 'on_{mode}_batch_end'.format(mode=mode)
    if self._check_timing and batch >= 1:
        batch_time = time.time() - self._batch_start_time
        self._batch_times.append(batch_time)
    self._call_batch_hook_helper(hook_name, batch, logs)
    if len(self._batch_times) >= self._num_batches_for_timing_check:
        end_hook_name = hook_name
        begin_hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
        avg_batch_time = sum(self._batch_times) / len(self._batch_times)
        avg_end_hook_time = sum(self._hook_times[end_hook_name]) / len(self._hook_times[end_hook_name])
        avg_begin_hook_time = sum(self._hook_times[begin_hook_name]) / len(self._hook_times[begin_hook_name])
        threshold_time = 1.0 * avg_batch_time
        warning_msg = 'Callback method `{hook}` is slow compared to the batch time (batch time: {batch_time:.4f}s vs `{hook}` time: {hook_time:.4f}s). Check your callbacks.'
        if avg_begin_hook_time > threshold_time:
            logging.warning(warning_msg.format(hook=begin_hook_name, batch_time=avg_batch_time, hook_time=avg_begin_hook_time))
        if avg_end_hook_time > threshold_time:
            logging.warning(warning_msg.format(hook=end_hook_name, batch_time=avg_batch_time, hook_time=avg_end_hook_time))
        self._check_timing = False
        self._batch_start_time = None
        self._batch_times = []
        self._hook_times = {}