import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
def tpu_encode(ts):
    """Encodes a nest of Tensors in a suitable way for TPUs.

  TPUs do not support tf.uint8, tf.uint16 and other data types. Furthermore,
  the speed of transfer and device reshapes depend on the shape of the data.
  This function tries to optimize the data encoding for a number of use cases.

  Should be used on CPU before sending data to TPU and in conjunction with
  `tpu_decode` after the data is transferred.

  Args:
    ts: A tf.nest of Tensors.

  Returns:
    A tf.nest of encoded Tensors.
  """

    def visit(t):
        num_elements = t.shape.num_elements()
        if t.dtype == tf.uint8 and num_elements is not None and (num_elements % 128 == 0):
            x = tf.xla.experimental.compile(lambda x: tf.transpose(x, list(range(1, t.shape.rank)) + [0]), [t])[0]
            x = tf.reshape(x, [-1, 4])
            x = tf.bitcast(x, tf.uint32)
            x = tf.reshape(x, [-1])
            return TPUEncodedUInt8(x, t.shape)
        elif t.dtype == tf.uint8:
            logging.warning('Inefficient uint8 transfer with shape: %s', t.shape)
            return tf.cast(t, tf.bfloat16)
        elif t.dtype == tf.uint16:
            return tf.cast(t, tf.int32)
        elif t.dtype == tf.float32 and t.shape.rank > 1 and (not (num_divisible(t.shape.dims, 128) >= 1 and num_divisible(t.shape.dims, 8) >= 2)):
            x = tf.reshape(t, [-1])
            return TPUEncodedF32(x, t.shape)
        else:
            return t
    return tf.nest.map_structure(visit, ts)