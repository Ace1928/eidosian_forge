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
@property
def unroll_specs(self):
    return tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape[1:], v.dtype), self._state)