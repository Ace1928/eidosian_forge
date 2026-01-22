from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import gc
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _loss_smaller(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is smaller.

  Both evaluation results should have the values for MetricKeys.LOSS, which are
  used for comparison.

  Args:
    best_eval_result: best eval metrics.
    current_eval_result: current eval metrics.

  Returns:
    True if the loss of current_eval_result is smaller; otherwise, False.

  Raises:
    ValueError: If input eval result is None or no loss is available.
  """
    default_key = metric_keys.MetricKeys.LOSS
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError('best_eval_result cannot be empty or no loss is found in it.')
    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError('current_eval_result cannot be empty or no loss is found in it.')
    return best_eval_result[default_key] > current_eval_result[default_key]