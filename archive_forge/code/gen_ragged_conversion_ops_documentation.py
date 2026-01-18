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
Helper used to compute the gradient for `RaggedTensorToVariant`.

  Computes the gradient for the dense_values input to the RaggedTensorToVariant
  op, given the variant-encoded ragged gradients of the outputs, along with
  the outer row-splits and the shape of the dense-values that were provided as
  inputs to the RaggedTensorToVariant op.

  Args:
    encoded_ragged_grad: A `Tensor` of type `variant`.
      A `variant` Tensor containing encoded `RaggedTensor` gradients.
    row_splits: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Outermost row-splits that were used as input to the RaggedTensorToVariant op.
    dense_values_shape: A `Tensor` of type `int32`.
      Shape of the dense_values that was used as an input to the
      RaggedTensorToVariant op.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tvalues`.
  