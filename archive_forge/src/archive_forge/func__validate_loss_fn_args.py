from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _validate_loss_fn_args(loss_fn):
    """Validates loss_fn arguments.

  Required arguments: labels, logits.
  Optional arguments: features.

  Args:
    loss_fn: The loss function.

  Raises:
    ValueError: If the signature is unexpected.
  """
    loss_fn_args = function_utils.fn_args(loss_fn)
    for required_arg in ['labels', 'logits']:
        if required_arg not in loss_fn_args:
            raise ValueError('loss_fn must contain argument: {}. Given arguments: {}'.format(required_arg, loss_fn_args))
    invalid_args = list(set(loss_fn_args) - set(['labels', 'logits', 'features']))
    if invalid_args:
        raise ValueError('loss_fn has unexpected args: {}'.format(invalid_args))