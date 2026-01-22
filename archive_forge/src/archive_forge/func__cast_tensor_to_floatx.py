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
def _cast_tensor_to_floatx(x):
    """Cast tensor to keras's floatx dtype if it is not already the same dtype."""
    if x.dtype == tf.keras.backend.floatx():
        return x
    else:
        return tf.cast(x, tf.keras.backend.floatx())