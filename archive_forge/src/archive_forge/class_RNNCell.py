import collections
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.keras.legacy_tf_layers import base as base_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['nn.rnn_cell.RNNCell'])
class RNNCell(base_layer.Layer):
    """Abstract object representing an RNN cell.

  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.  The optional
  third input argument, `scope`, is allowed for backwards compatibility
  purposes; but should be left off for new subclasses.

  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

    def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
        super(RNNCell, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._is_tf_rnn_cell = True

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple with
        shapes `[batch_size, s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
        if scope is not None:
            with vs.variable_scope(scope, custom_getter=self._rnn_get_variable) as scope:
                return super(RNNCell, self).__call__(inputs, state, scope=scope)
        else:
            scope_attrname = 'rnncell_scope'
            scope = getattr(self, scope_attrname, None)
            if scope is None:
                scope = vs.variable_scope(vs.get_variable_scope(), custom_getter=self._rnn_get_variable)
                setattr(self, scope_attrname, scope)
            with scope:
                return super(RNNCell, self).__call__(inputs, state)

    def _rnn_get_variable(self, getter, *args, **kwargs):
        variable = getter(*args, **kwargs)
        if ops.executing_eagerly_outside_functions():
            trainable = variable.trainable
        else:
            trainable = variable in tf_variables.trainable_variables() or (base_layer_utils.is_split_variable(variable) and list(variable)[0] in tf_variables.trainable_variables())
        if trainable and all((variable is not v for v in self._trainable_weights)):
            self._trainable_weights.append(variable)
        elif not trainable and all((variable is not v for v in self._non_trainable_weights)):
            self._non_trainable_weights.append(variable)
        return variable

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
        raise NotImplementedError('Abstract method')

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError('Abstract method')

    def build(self, _):
        pass

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            inputs = tensor_conversion.convert_to_tensor_v2_with_dispatch(inputs, name='inputs')
            if batch_size is not None:
                if tensor_util.is_tf_type(batch_size):
                    static_batch_size = tensor_util.constant_value(batch_size, partial=True)
                else:
                    static_batch_size = batch_size
                if inputs.shape.dims[0].value != static_batch_size:
                    raise ValueError('batch size from input tensor is different from the input param. Input tensor batch: {}, batch_size: {}'.format(inputs.shape.dims[0].value, batch_size))
            if dtype is not None and inputs.dtype != dtype:
                raise ValueError('dtype from input tensor is different from the input param. Input tensor dtype: {}, dtype: {}'.format(inputs.dtype, dtype))
            batch_size = inputs.shape.dims[0].value or array_ops.shape(inputs)[0]
            dtype = inputs.dtype
        if batch_size is None or dtype is None:
            raise ValueError('batch_size and dtype cannot be None while constructing initial state: batch_size={}, dtype={}'.format(batch_size, dtype))
        return self.zero_state(batch_size, dtype)

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size, s]` for each s in `state_size`.
    """
        state_size = self.state_size
        is_eager = context.executing_eagerly()
        if is_eager and _hasattr(self, '_last_zero_state'):
            last_state_size, last_batch_size, last_dtype, last_output = getattr(self, '_last_zero_state')
            if last_batch_size == batch_size and last_dtype == dtype and (last_state_size == state_size):
                return last_output
        with backend.name_scope(type(self).__name__ + 'ZeroState'):
            output = _zero_state_tensors(state_size, batch_size, dtype)
        if is_eager:
            self._last_zero_state = (state_size, batch_size, dtype, output)
        return output

    def get_config(self):
        return super(RNNCell, self).get_config()

    @property
    def _use_input_spec_as_call_signature(self):
        return False