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
class _MultiIOSubclassModelCustomBuild(models.Model):
    """Multi IO Keras subclass model that uses a custom build method."""

    def __init__(self, branch_a_func, branch_b_func, shared_input_branch_func=None, shared_output_branch_func=None):
        super(_MultiIOSubclassModelCustomBuild, self).__init__()
        self._shared_input_branch_func = shared_input_branch_func
        self._branch_a_func = branch_a_func
        self._branch_b_func = branch_b_func
        self._shared_output_branch_func = shared_output_branch_func
        self._shared_input_branch = None
        self._branch_a = None
        self._branch_b = None
        self._shared_output_branch = None

    def build(self, input_shape):
        if self._shared_input_branch_func():
            self._shared_input_branch = self._shared_input_branch_func()
        self._branch_a = self._branch_a_func()
        self._branch_b = self._branch_b_func()
        if self._shared_output_branch_func():
            self._shared_output_branch = self._shared_output_branch_func()

    def call(self, inputs, **kwargs):
        if self._shared_input_branch:
            for layer in self._shared_input_branch:
                inputs = layer(inputs)
            a = inputs
            b = inputs
        else:
            a, b = inputs
        for layer in self._branch_a:
            a = layer(a)
        for layer in self._branch_b:
            b = layer(b)
        outs = (a, b)
        if self._shared_output_branch:
            for layer in self._shared_output_branch:
                outs = layer(outs)
        return outs