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
def _create_or_get_iterations_per_loop():
    """Creates or gets the iterations_per_loop variable.

  In TPUEstimator, the user provided computation, the model_fn, is wrapped
  inside a tf.while_loop for peak performance. The iterations of the loop are
  specified by this variable, which adjusts its value on the CPU after each TPU
  program execution and before the next TPU execution.

  The purpose of using a variable, rather then a constant, is to allow
  TPUEstimator adapt the TPU training iterations according to the final steps
  specified by users. For example, if the user sets the iterations_per_loop as 4
  in TPUConfig and steps as 10 in TPUEstimator.train(), the iterations_per_loop
  variable will have the following value before each TPU training.

      - 1-th TPU execution: iterations_per_loop = 4
      - 2-th TPU execution: iterations_per_loop = 4
      - 3-th TPU execution: iterations_per_loop = 2

  As model_fn increases the global step once per train_op invocation, the global
  step is 10 after all TPU executions, matching the steps=10 inputs passed in by
  users.

  Returns:
    A TF non-trainable resource variable.

  Raises:
    RuntimeError: If multi iterations_per_loop variables were found.
  """
    graph = tf.compat.v1.get_default_graph()
    collection_name = '{}_{}'.format(_TPU_ESTIMATOR, _ITERATIONS_PER_LOOP_VAR)
    iter_vars = graph.get_collection(collection_name)
    if len(iter_vars) == 1:
        return iter_vars[0]
    elif len(iter_vars) > 1:
        raise RuntimeError('Multiple iterations_per_loop_var in collection.')
    with ops.colocate_with(tf.compat.v1.train.get_global_step()):
        with tf.compat.v1.variable_scope(_TPU_ESTIMATOR, reuse=tf.compat.v1.AUTO_REUSE):
            return tf.compat.v1.get_variable(_ITERATIONS_PER_LOOP_VAR, initializer=tf.compat.v1.initializers.zeros(), shape=[], dtype=tf.dtypes.int32, trainable=False, collections=[collection_name, tf.compat.v1.GraphKeys.LOCAL_VARIABLES], use_resource=True)