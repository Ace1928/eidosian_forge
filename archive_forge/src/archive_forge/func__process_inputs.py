import collections
import warnings
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _process_inputs(self, inputs, initial_state, constants):
    if isinstance(inputs, collections.abc.Sequence) and (not isinstance(inputs, tuple)):
        if not self._num_constants:
            initial_state = inputs[1:]
        else:
            initial_state = inputs[1:-self._num_constants]
            constants = inputs[-self._num_constants:]
        if len(initial_state) == 0:
            initial_state = None
        inputs = inputs[0]
    if self.stateful:
        if initial_state is not None:
            non_zero_count = math_ops.add_n([math_ops.count_nonzero_v2(s) for s in nest.flatten(self.states)])
            initial_state = cond.cond(non_zero_count > 0, true_fn=lambda: self.states, false_fn=lambda: initial_state, strict=True)
        else:
            initial_state = self.states
    elif initial_state is None:
        initial_state = self.get_initial_state(inputs)
    if len(initial_state) != len(self.states):
        raise ValueError('Layer has ' + str(len(self.states)) + ' states but was passed ' + str(len(initial_state)) + ' initial states.')
    return (inputs, initial_state, constants)