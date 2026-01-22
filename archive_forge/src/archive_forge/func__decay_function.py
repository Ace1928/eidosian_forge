import abc
import math
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization as legacy_serialization
from tensorflow.python.util.tf_export import keras_export
def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
    with tf.name_scope(self.name or 'CosineDecay'):
        completed_fraction = step / decay_steps
        tf_pi = tf.constant(math.pi, dtype=dtype)
        cosine_decayed = 0.5 * (1.0 + tf.cos(tf_pi * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha
        return tf.multiply(decay_from_lr, decayed)