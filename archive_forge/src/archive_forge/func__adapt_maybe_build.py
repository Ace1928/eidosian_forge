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
def _adapt_maybe_build(self, data):
    if not self.built:
        try:
            data_shape = data.shape
            data_shape_nones = tuple([None] * len(data.shape))
        except AttributeError:
            data_shape = None
            data_shape_nones = None
        batch_input_shape = getattr(self, '_batch_input_shape', None)
        if batch_input_shape is None:
            self._batch_input_shape = data_shape_nones
        self.build(data_shape)
        self.built = True