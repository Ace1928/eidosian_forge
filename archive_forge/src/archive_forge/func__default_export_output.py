from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _default_export_output(export_outputs, head_name):
    """Extracts the default export output from the given export_outputs dict."""
    if len(export_outputs) == 1:
        return next(six.itervalues(export_outputs))
    try:
        return export_outputs[base_head.DEFAULT_SERVING_KEY]
    except KeyError:
        raise ValueError('{} did not specify default export_outputs. Given: {} Suggested fix: Use one of the heads in tf.estimator, or include key {} in export_outputs.'.format(head_name, export_outputs, base_head.DEFAULT_SERVING_KEY))