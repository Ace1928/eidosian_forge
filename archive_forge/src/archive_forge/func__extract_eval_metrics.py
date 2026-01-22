from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import six
import tensorflow as tf
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_constants
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _extract_eval_metrics(output_dict):
    """Return a eval metric dict extracted from the output_dict.

  Eval metrics consist of a value tensor and an update op. Both must be in the
  passed-in tensor dictionary for an eval metric to be added to the returned
  dictionary.

  Args:
    output_dict: a dict that maps strings to tensors.

  Returns:
    dict mapping strings to (value, update_op) tuples.
  """
    metric_ops = {}
    separator_char = export_lib._SupervisedOutput._SEPARATOR_CHAR
    for key, tensor in six.iteritems(output_dict):
        split_key = key.split(separator_char)
        metric_name = separator_char.join(split_key[:-1])
        if split_key[0] == export_lib._SupervisedOutput.METRICS_NAME:
            if split_key[-1] == export_lib._SupervisedOutput.METRIC_VALUE_SUFFIX:
                update_op = ''.join([metric_name, separator_char, export_lib._SupervisedOutput.METRIC_UPDATE_SUFFIX])
                if update_op in output_dict:
                    update_op_tensor = output_dict[update_op]
                    metric_ops[metric_name] = (tensor, update_op_tensor)
    return metric_ops