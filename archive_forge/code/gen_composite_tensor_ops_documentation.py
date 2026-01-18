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
Decodes a `variant` scalar Tensor into an `ExtensionType` value.

  Returns the Tensor components encoded in a `CompositeTensorVariant`.

  Raises an error if `type_spec_proto` doesn't match the TypeSpec
  in `encoded`.

  Args:
    encoded: A `Tensor` of type `variant`.
      A scalar `variant` Tensor containing an encoded ExtensionType value.
    metadata: A `string`.
      String serialization for the TypeSpec.  Must be compatible with the
      `TypeSpec` contained in `encoded`.  (Note: the encoding for the TypeSpec
      may change in future versions of TensorFlow.)
    Tcomponents: A list of `tf.DTypes`. Expected dtypes for components.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tcomponents`.
  