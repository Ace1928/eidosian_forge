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
def _any_weight_initialized(keras_model):
    """Check if any weights has been initialized in the Keras model.

  Args:
    keras_model: An instance of compiled keras model.

  Returns:
    boolean, True if at least one weight has been initialized, else False.
    Currently keras initialize all weights at get_session().
  """
    if keras_model is None:
        return False
    if tf.compat.v1.executing_eagerly_outside_functions():
        return True
    for layer in keras_model.layers:
        for weight in layer.weights:
            if hasattr(weight, '_keras_initialized'):
                return True
    return False