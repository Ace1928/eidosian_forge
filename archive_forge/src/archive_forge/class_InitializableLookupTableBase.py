import collections
import functools
import uuid
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_lookup_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import compat as compat_util
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
class InitializableLookupTableBase(LookupInterface):
    """Initializable lookup table interface.

  An initializable lookup tables persist across different steps.
  """

    def __init__(self, default_value, initializer):
        """Construct a table object from a table reference.

    If requires a table initializer object (subclass of `TableInitializerBase`).
    It provides the table key and value types, as well as the op to initialize
    the table. The caller is responsible to execute the initialization op.

    Args:
      default_value: The value to use if a key is missing in the table.
      initializer: The table initializer to use.
    """
        super(InitializableLookupTableBase, self).__init__(initializer.key_dtype, initializer.value_dtype)
        self._default_value = ops.convert_to_tensor(default_value, dtype=self._value_dtype)
        self._default_value.get_shape().merge_with(tensor_shape.TensorShape([]))
        if isinstance(initializer, trackable_base.Trackable):
            self._initializer = self._track_trackable(initializer, '_initializer')
        with ops.init_scope():
            self._resource_handle = self._create_resource()
        if not context.executing_eagerly() and ops.get_default_graph()._get_control_flow_context() is not None:
            with ops.init_scope():
                self._init_op = self._initialize()
        else:
            self._init_op = self._initialize()

    def _initialize(self):
        return self._initializer.initialize(self)

    @property
    def default_value(self):
        """The default value of the table."""
        return self._default_value

    def size(self, name=None):
        """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
        with ops.name_scope(name, '%s_Size' % self.name, [self.resource_handle]):
            return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

    def lookup(self, keys, name=None):
        """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: A name for the operation (optional).

    Returns:
      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,
      otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` or `default_value` doesn't match the table data
        types.
    """
        key_tensor = keys
        if isinstance(keys, (sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor)):
            key_tensor = keys.values
        if keys.dtype.base_dtype != self._key_dtype:
            raise TypeError(f'Dtype of argument `keys` must be {self._key_dtype}, received: {keys.dtype}')
        with ops.name_scope(name, '%s_Lookup' % self.name, (self.resource_handle, key_tensor, self._default_value)):
            values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, key_tensor, self._default_value)
        values.set_shape(key_tensor.get_shape())
        if isinstance(keys, sparse_tensor.SparseTensor):
            return sparse_tensor.SparseTensor(keys.indices, values, keys.dense_shape)
        elif isinstance(keys, ragged_tensor.RaggedTensor):
            return keys.with_values(values)
        else:
            return values