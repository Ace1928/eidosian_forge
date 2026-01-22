import abc
import platform
import re
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src import initializers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def _create_iteration_variable(self):
    if self._mesh:
        init_val = tf.constant(0, dtype=tf.int64)
        init_val = tf.experimental.dtensor.copy_to_mesh(init_val, tf.experimental.dtensor.Layout.replicated(self._mesh, rank=0))
        with tf.init_scope():
            self._iterations = tf.experimental.dtensor.DVariable(init_val, name='iteration')
        self._variables.append(self._iterations)
    else:
        super()._create_iteration_variable()