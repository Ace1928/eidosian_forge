from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import enum
import math
import os
import signal
import sys
import threading
import time
import tensorflow as tf
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import preempted_hook
from tensorflow.python.tpu import session_support
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import evaluation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import error_handling
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_context
from tensorflow_estimator.python.estimator.tpu import util as util_lib
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdagradParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdamParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import EmbeddingConfigSpec  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import StochasticGradientDescentParameters  # pylint: disable=unused-import
class BatchConfig(collections.namedtuple('BatchConfig', ['num_batch_threads', 'max_batch_size', 'batch_timeout_micros', 'allowed_batch_sizes', 'max_enqueued_batches'])):
    """Class to handle config inputs into the batching function."""

    def __new__(cls, num_batch_threads, max_batch_size, batch_timeout_micros, allowed_batch_sizes, max_enqueued_batches=100):
        """Creates an BatchConfig instance.

    Args:
     num_batch_threads: Number of scheduling threads for processing batches of
       work. Determines the number of batches processed in parallel.
      max_batch_size: Batch sizes will never be bigger than this.
      batch_timeout_micros: Maximum number of microseconds to wait before
        outputting an incomplete batch.
      allowed_batch_sizes: Optional list of allowed batch sizes. If left empty,
        does nothing. Otherwise, supplies a list of batch sizes, causing the op
        to pad batches up to one of those sizes. The entries must increase
        monotonically, and the final entry must equal max_batch_size.
      max_enqueued_batches: The maximum depth of the batch queue. Defaults to
        100.

    Returns:
      An BatchConfig instance.
    """
        return super(BatchConfig, cls).__new__(cls, num_batch_threads=num_batch_threads, max_batch_size=max_batch_size, batch_timeout_micros=batch_timeout_micros, allowed_batch_sizes=allowed_batch_sizes, max_enqueued_batches=max_enqueued_batches)