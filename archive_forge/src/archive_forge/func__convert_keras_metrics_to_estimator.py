from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _convert_keras_metrics_to_estimator(model, metric_names_map=None):
    """Convert metrics from a Keras model to ops used by the Estimator framework.

  Args:
    model: A `tf.keras.Model` object.
    metric_names_map: Optional dictionary mapping Keras model output metric
      names to custom names.

  Returns:
    Dictionary mapping metric names to tuples of (value, update) ops. May return
    `None` if the model does not contain any metrics.
  """
    if not getattr(model, '_compile_metrics', None):
        return None
    compiled_metrics = model._compile_metric_functions
    if metric_names_map:
        custom_map_keys = set(metric_names_map.keys())
        expected_keys = {m.name for m in compiled_metrics}
        unknown = expected_keys.difference(custom_map_keys)
        if unknown:
            raise ValueError('Invalid `metric_names_map`. The following keras model metric names:"{}" do not exist in the `metric_names_map` dictionary'.format(list(unknown)))
        extra = custom_map_keys.difference(expected_keys)
        if extra:
            raise ValueError('Invalid `metric_names_map`. There are unexpected keys in the `metric_names_map` dictionary. Expected keys: {}, Received: {}'.format(list(expected_keys), list(extra)))
        return {metric_names_map[m.name]: m for m in compiled_metrics}
    else:
        return {m.name: m for m in compiled_metrics}