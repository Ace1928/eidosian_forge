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
def _write_dict_to_summary(output_dir, dictionary, current_global_step):
    """Writes a `dict` into summary file in given output directory.

  Args:
    output_dir: `str`, directory to write the summary file in.
    dictionary: the `dict` to be written to summary file.
    current_global_step: `int`, the current global step.
  """
    tf.compat.v1.logging.info('Saving dict for global step %d: %s', current_global_step, _dict_to_str(dictionary))
    summary_writer = tf.compat.v1.summary.FileWriterCache.get(output_dir)
    summary_proto = summary_pb2.Summary()
    for key in dictionary:
        if dictionary[key] is None:
            continue
        if key == 'global_step':
            continue
        if isinstance(dictionary[key], np.float32) or isinstance(dictionary[key], float):
            summary_proto.value.add(tag=key, simple_value=float(dictionary[key]))
        elif isinstance(dictionary[key], np.int64) or isinstance(dictionary[key], np.int32) or isinstance(dictionary[key], int):
            summary_proto.value.add(tag=key, simple_value=int(dictionary[key]))
        elif isinstance(dictionary[key], six.binary_type):
            try:
                summ = summary_pb2.Summary.FromString(dictionary[key])
                for i, _ in enumerate(summ.value):
                    summ.value[i].tag = '%s/%d' % (key, i)
                summary_proto.value.extend(summ.value)
            except message.DecodeError:
                tf.compat.v1.logging.warn('Skipping summary for %s, cannot parse string to Summary.', key)
                continue
        elif isinstance(dictionary[key], np.ndarray):
            value = summary_proto.value.add()
            value.tag = key
            value.node_name = key
            tensor_proto = tf.make_tensor_proto(dictionary[key])
            value.tensor.CopyFrom(tensor_proto)
            tf.compat.v1.logging.info('Summary for np.ndarray is not visible in Tensorboard by default. Consider using a Tensorboard plugin for visualization (see https://github.com/tensorflow/tensorboard-plugin-example/blob/master/README.md for more information).')
        else:
            tf.compat.v1.logging.warn('Skipping summary for %s, must be a float, np.float32, np.int64, np.int32 or int or np.ndarray or a serialized string of Summary.', key)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()