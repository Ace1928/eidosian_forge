import collections
import contextlib
import functools
import itertools
import threading
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator
class _SmallSubclassMLPCustomBuild(models.Model):
    """A subclass model small MLP that uses a custom build method."""

    def __init__(self, num_hidden, num_classes):
        super(_SmallSubclassMLPCustomBuild, self).__init__()
        self.layer_a = None
        self.layer_b = None
        self.num_hidden = num_hidden
        self.num_classes = num_classes

    def build(self, input_shape):
        self.layer_a = layers.Dense(self.num_hidden, activation='relu')
        activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
        self.layer_b = layers.Dense(self.num_classes, activation=activation)

    def call(self, inputs, **kwargs):
        x = self.layer_a(inputs)
        return self.layer_b(x)