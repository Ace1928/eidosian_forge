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
class TPUEncodedF32(composite_tensor.CompositeTensor):

    def __init__(self, encoded, shape):
        self.encoded = encoded
        self.original_shape = shape
        self._spec = TPUEncodedF32Spec(encoded.shape, tf.TensorShape(shape))

    @property
    def _type_spec(self):
        return self._spec