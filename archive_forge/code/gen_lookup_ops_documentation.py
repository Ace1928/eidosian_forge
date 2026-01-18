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
Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  