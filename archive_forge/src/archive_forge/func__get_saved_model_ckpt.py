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
def _get_saved_model_ckpt(saved_model_dir):
    """Return path to variables checkpoint in a `SavedModel` directory."""
    if not tf.compat.v1.gfile.Exists(os.path.join(path_helpers.get_variables_dir(saved_model_dir), tf.compat.as_text('variables.index'))):
        raise ValueError('Directory provided has an invalid SavedModel format: %s' % saved_model_dir)
    return path_helpers.get_variables_path(saved_model_dir)