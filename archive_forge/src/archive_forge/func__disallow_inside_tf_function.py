import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable
def _disallow_inside_tf_function(method_name):
    """Disallow calling a method inside a `tf.function`."""
    if ops.inside_function():
        error_msg = 'Detected a call to `PreprocessingLayer.{method_name}` inside a `tf.function`. `PreprocessingLayer.{method_name} is a high-level endpoint that manages its own `tf.function`. Please move the call to `PreprocessingLayer.{method_name}` outside of all enclosing `tf.function`s. Note that you can call a `PreprocessingLayer` directly on `Tensor`s inside a `tf.function` like: `layer(x)`, or update its state like: `layer.update_state(x)`.'.format(method_name=method_name)
        raise RuntimeError(error_msg)