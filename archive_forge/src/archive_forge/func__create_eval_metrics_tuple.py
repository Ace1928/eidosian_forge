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
def _create_eval_metrics_tuple(fn, kwargs):
    """Creates TPU eval metrics tuple.

  Helper function to make eval_metric tuple (eval_metric_fn, fn_kwargs) used
  by `TPUEstimator`. TPUEstimator requires that `eval_metric_fn` take
  exclusively Tensor arguments. This helper can help create such a function from
  a more generic function that can take both Tensor and non-Tensor arguments.

  Args:
    fn: A eval_metric_fn that takes both Tensor and non-Tensor arguments. This
      function must return a dict of form
        {'metric name': (metric_tensor, eval_op)}
    kwargs: Dict of arguments for `fn`.

  Returns:
    `eval_metric` tuple that can be passed to a `model_fn._TPUEstimatorSpec`.
  """
    tensor_kwargs = {}
    nontensor_kwargs = {}
    for k, v in six.iteritems(kwargs):
        if tf.is_tensor(v):
            tensor_kwargs[k] = v
        else:
            nontensor_kwargs[k] = v

    def _fn(**tensors):
        return fn(**dict(nontensor_kwargs, **tensors))
    return (_fn, tensor_kwargs)