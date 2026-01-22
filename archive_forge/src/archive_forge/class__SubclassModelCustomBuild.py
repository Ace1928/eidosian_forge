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
class _SubclassModelCustomBuild(models.Model):
    """A Keras subclass model that uses a custom build method."""

    def __init__(self, layer_generating_func, *args, **kwargs):
        super(_SubclassModelCustomBuild, self).__init__(*args, **kwargs)
        self.all_layers = None
        self._layer_generating_func = layer_generating_func

    def build(self, input_shape):
        model_layers = []
        for layer in self._layer_generating_func():
            model_layers.append(layer)
        self.all_layers = model_layers

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.all_layers:
            x = layer(x)
        return x