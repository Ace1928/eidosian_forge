from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import os
import tempfile
import numpy as np
import six
import tensorflow as tf
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.tools.docs import doc_controls
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _write_checkpoint_path_to_summary(output_dir, checkpoint_path, current_global_step):
    """Writes `checkpoint_path` into summary file in the given output directory.

  Args:
    output_dir: `str`, directory to write the summary file in.
    checkpoint_path: `str`, checkpoint file path to be written to summary file.
    current_global_step: `int`, the current global step.
  """
    checkpoint_path_tag = 'checkpoint_path'
    tf.compat.v1.logging.info("Saving '%s' summary for global step %d: %s", checkpoint_path_tag, current_global_step, checkpoint_path)
    summary_proto = summary_pb2.Summary()
    summary_proto.value.add(tag=checkpoint_path_tag, tensor=tf.make_tensor_proto(checkpoint_path, dtype=tf.dtypes.string))
    summary_writer = tf.compat.v1.summary.FileWriterCache.get(output_dir)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()