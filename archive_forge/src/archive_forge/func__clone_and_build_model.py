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
def _clone_and_build_model(mode, keras_model, custom_objects, features=None, labels=None, optimizer_config=None):
    """Clone and build the given keras_model.

  Args:
    mode: training mode.
    keras_model: an instance of compiled keras model.
    custom_objects: Dictionary for custom objects.
    features: Dict of tensors.
    labels: Dict of tensors, or single tensor instance.
    optimizer_config: Optimizer config dictionary, returned by
      `optimizer.get_config()`. This is used when cloning a model with an
      optimizer. Since `_clone_and_build_model` is called in a different graph
      and session from the model, `optimizer.get_config()` may raise an error
      during the attempt to serialize the optimizer hyperparameter values.

  Returns:
    The newly built model.
  """
    tf.keras.backend.set_learning_phase(mode == ModeKeys.TRAIN)
    input_tensors, target_tensors, sample_weight_tensors = _convert_estimator_io_to_keras(keras_model, features, labels)
    compile_clone = mode != ModeKeys.PREDICT
    global_step = None
    if compile_clone:
        global_step = tf.compat.v1.train.get_or_create_global_step()
        tf.compat.v2.keras.__internal__.backend.track_variable(global_step)
    clone = tf.compat.v2.keras.__internal__.models.clone_and_build_model(keras_model, input_tensors, target_tensors, custom_objects, compile_clone=compile_clone, in_place_reset=not keras_model._is_graph_network, optimizer_iterations=global_step, optimizer_config=optimizer_config)
    if sample_weight_tensors is not None:
        sample_weight_tensors = standardize_sample_weights(sample_weight_tensors, clone.output_names)
        clone._compile_weights_loss_and_weighted_metrics(sample_weight_tensors)
    return clone