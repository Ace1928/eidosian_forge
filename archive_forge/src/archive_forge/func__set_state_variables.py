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
def _set_state_variables(self, updates):
    """Directly update the internal state of this Layer.

    This method expects a string-keyed dict of {state_variable_name: state}. The
    precise nature of the state, and the names associated, are describe by
    the subclasses of CombinerPreprocessingLayer.

    Args:
      updates: A string keyed dict of weights to update.

    Raises:
      RuntimeError: if 'build()' was not called before 'set_processing_state'.
    """
    if not self.built:
        raise RuntimeError('_set_state_variables() must be called after build().')
    with ops.init_scope():
        for var_name, value in updates.items():
            self.state_variables[var_name].assign(value)