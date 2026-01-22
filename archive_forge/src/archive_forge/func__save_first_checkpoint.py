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
def _save_first_checkpoint(keras_model, custom_objects, config, save_object_ckpt):
    """Save first checkpoint for the keras Estimator.

  Args:
    keras_model: an instance of compiled keras model.
    custom_objects: Dictionary for custom objects.
    config: Estimator config.
    save_object_ckpt: Whether to save an object-based checkpoint.

  Returns:
    The path where keras model checkpoint is saved.
  """
    keras_model_dir = os.path.join(config.model_dir, 'keras')
    latest_path = tf.train.latest_checkpoint(keras_model_dir)
    if not latest_path:
        keras_weights = None
        if _any_weight_initialized(keras_model):
            keras_weights = keras_model.get_weights()
        if not tf.compat.v1.gfile.IsDirectory(keras_model_dir):
            tf.compat.v1.gfile.MakeDirs(keras_model_dir)
        with tf.Graph().as_default():
            tf.compat.v1.random.set_random_seed(config.tf_random_seed)
            tf.compat.v1.train.create_global_step()
            model = _clone_and_build_model(ModeKeys.TRAIN, keras_model, custom_objects)
            model._make_train_function()
            with tf.compat.v1.Session(config=config.session_config) as sess:
                if keras_weights:
                    model.set_weights(keras_weights)
                tf.compat.v2.keras.__internal__.backend.initialize_variables(sess)
                if save_object_ckpt:
                    model._track_trackable(tf.compat.v1.train.get_global_step(), 'estimator_global_step')
                    latest_path = os.path.join(keras_model_dir, 'keras_model.ckpt')
                    model.save_weights(latest_path)
                else:
                    saver = tf.compat.v1.train.Saver()
                    latest_path = os.path.join(keras_model_dir, 'keras_model.ckpt')
                    saver.save(sess, latest_path)
    return latest_path