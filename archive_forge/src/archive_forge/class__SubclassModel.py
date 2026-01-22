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
class _SubclassModel(models.Model):
    """A Keras subclass model."""

    def __init__(self, model_layers, *args, **kwargs):
        """Instantiate a model.

    Args:
      model_layers: a list of layers to be added to the model.
      *args: Model's args
      **kwargs: Model's keyword args, at most one of input_tensor -> the input
        tensor required for ragged/sparse input.
    """
        inputs = kwargs.pop('input_tensor', None)
        super(_SubclassModel, self).__init__(*args, **kwargs)
        for i, layer in enumerate(model_layers):
            setattr(self, self._layer_name_for_i(i), layer)
        self.num_layers = len(model_layers)
        if inputs is not None:
            self._set_inputs(inputs)

    def _layer_name_for_i(self, i):
        return 'layer{}'.format(i)

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(self.num_layers):
            layer = getattr(self, self._layer_name_for_i(i))
            x = layer(x)
        return x