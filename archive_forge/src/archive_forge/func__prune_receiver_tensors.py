from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.eager import function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import func_graph
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _prune_receiver_tensors(wrapped_function, receiver_tensors, outputs, name):
    inputs = _canonicalize_receiver_tensors(receiver_tensors)
    return wrapped_function.prune(inputs, outputs, name=name, input_signature=(None, func_graph.convert_structure_to_signature(inputs)))