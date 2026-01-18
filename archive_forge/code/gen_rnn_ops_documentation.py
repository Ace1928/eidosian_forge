import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
Computes the LSTM cell backward propagation for 1 timestep.

  This implementation is to be used in conjunction of LSTMBlockCell.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The input to the LSTM cell, shape (batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      The previous cell state.
    h_prev: A `Tensor`. Must have the same type as `x`. The previous h state.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    i: A `Tensor`. Must have the same type as `x`. The input gate.
    cs: A `Tensor`. Must have the same type as `x`.
      The cell state before the tanh.
    f: A `Tensor`. Must have the same type as `x`. The forget gate.
    o: A `Tensor`. Must have the same type as `x`. The output gate.
    ci: A `Tensor`. Must have the same type as `x`. The cell input.
    co: A `Tensor`. Must have the same type as `x`. The cell after the tanh.
    cs_grad: A `Tensor`. Must have the same type as `x`.
      The current gradient of cs.
    h_grad: A `Tensor`. Must have the same type as `x`.
      The gradient of h vector.
    use_peephole: A `bool`. Whether the cell uses peephole connections.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (cs_prev_grad, dicfo, wci_grad, wcf_grad, wco_grad).

    cs_prev_grad: A `Tensor`. Has the same type as `x`.
    dicfo: A `Tensor`. Has the same type as `x`.
    wci_grad: A `Tensor`. Has the same type as `x`.
    wcf_grad: A `Tensor`. Has the same type as `x`.
    wco_grad: A `Tensor`. Has the same type as `x`.
  