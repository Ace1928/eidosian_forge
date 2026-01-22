from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import os
import tempfile
import numpy as np
import six
import tensorflow as tf
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.tools.docs import doc_controls
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _train_model_distributed(self, input_fn, hooks, saving_listeners):
    """Initiate training with `input_fn`, using `DistributionStrategies`.

    Args:
      input_fn: A function that provides input data for training as minibatches.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      saving_listeners: list of `tf.train.CheckpointSaverListener` objects. Used
        for callbacks that run immediately before or after checkpoint savings.

    Returns:
      Loss from training
    """
    if hasattr(self._config, '_distribute_coordinator_mode') and self._config._distribute_coordinator_mode:
        distribute_coordinator_training.estimator_train(self, lambda est, s, train_hooks: est._actual_train_model_distributed(s, input_fn, train_hooks, saving_listeners), hooks)
        return self
    else:
        self._config._train_distribute.configure(self._config.session_config)
        return self._actual_train_model_distributed(self._config._train_distribute, input_fn, hooks, saving_listeners)