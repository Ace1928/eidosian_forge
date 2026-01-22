from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.saved_model import model_utils as export_utils
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _update_variable_collection(collection_name, vars_to_add):
    """Add variables to collection."""
    collection = set(tf.compat.v1.get_collection(collection_name))
    vars_to_add = vars_to_add.difference(collection)
    for v in vars_to_add:
        tf.compat.v1.add_to_collection(collection_name, v)