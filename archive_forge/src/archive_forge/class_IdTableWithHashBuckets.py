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
class IdTableWithHashBuckets(LookupInterface):
    """String to Id table wrapper that assigns out-of-vocabulary keys to buckets.

  For example, if an instance of `IdTableWithHashBuckets` is initialized with a
  string-to-id table that maps:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`

  The `IdTableWithHashBuckets` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where bucket_id will be between `3` and
  `3 + num_oov_buckets - 1`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is `["emerson", "lake", "palmer", "king", "crimson"]`,
  the lookup result is `[0, 1, 2, 4, 7]`.

  If `table` is None, only out-of-vocabulary buckets are used.

  Example usage:

  ```python
  num_oov_buckets = 3
  input_tensor = tf.constant(["emerson", "lake", "palmer", "king", "crimnson"])
  table = tf.IdTableWithHashBuckets(
      tf.StaticHashTable(
          tf.lookup.TextFileInitializer(
              filename,
              key_dtype=tf.string,
              key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
              value_dtype=tf.int64,
              value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
              delimiter="\\t"),
          default_value),
      num_oov_buckets)
  out = table.lookup(input_tensor).
  table.init.run()
  print(out.eval())
  ```

  The hash function used for generating out-of-vocabulary buckets ID is handled
  by `hasher_spec`.
  """

    def __init__(self, table, num_oov_buckets, hasher_spec=FastHashSpec, name=None, key_dtype=None):
        """Construct a `IdTableWithHashBuckets` object.

    Args:
      table: Table that maps `tf.string` or `tf.int64` keys to `tf.int64` ids.
      num_oov_buckets: Number of buckets to use for out-of-vocabulary keys.
      hasher_spec: A `HasherSpec` to specify the hash function to use for
        assignation of out-of-vocabulary buckets  (optional).
      name: A name for the operation (optional).
      key_dtype: Data type of keys passed to `lookup`. Defaults to
        `table.key_dtype` if `table` is specified, otherwise `tf.string`. Must
        be string or integer, and must be castable to `table.key_dtype`.

    Raises:
      ValueError: when `table` in None and `num_oov_buckets` is not positive.
      TypeError: when `hasher_spec` is invalid.
    """
        if name:
            name = name.rstrip('/')
        if table:
            if key_dtype is None:
                key_dtype = table.key_dtype
            supported_table_key_dtypes = (dtypes.int64, dtypes.string)
            if table.key_dtype not in supported_table_key_dtypes:
                raise TypeError(f'Invalid `key_dtype`, expected one of {supported_table_key_dtypes}, received {key_dtype}.')
            if table.key_dtype.is_integer != key_dtype.is_integer:
                raise TypeError('Invalid `key dtype`, expected %s but got %s.' % ('integer' if key_dtype.is_integer else 'non-integer', table.key_dtype))
            if table.value_dtype != dtypes.int64:
                raise TypeError('Invalid `value_dtype`: expected int64 but got %s.' % table.value_dtype)
            self._table = table
            name = name or self._table.name
        else:
            if num_oov_buckets <= 0:
                raise ValueError('`oov_buckets` must be > 0 if no `table` is supplied.')
            key_dtype = dtypes.string if key_dtype is None else key_dtype
            self._table = None
            name = name or 'hash_bucket'
        if not key_dtype.is_integer and dtypes.string != key_dtype:
            raise TypeError(f'Invalid `key_dtype`, expected integer or string, got {key_dtype}.')
        self._num_oov_buckets = num_oov_buckets
        if not isinstance(hasher_spec, HasherSpec):
            raise TypeError(f'`hasher_spec` must be of type HasherSpec, got {type(hasher_spec)}.')
        self._hasher_spec = hasher_spec
        if name:
            self._table_name = name.split('/')[-1]
        else:
            self._table_name = None
        super(IdTableWithHashBuckets, self).__init__(key_dtype, dtypes.int64)

    def _create_resource(self):
        if self._table is not None:
            return self._table._create_resource()
        return None

    def _initialize(self):
        if self._table is not None:
            return self._table._initialize()
        with ops.name_scope(None, 'init'):
            return control_flow_ops.no_op()

    @property
    def initializer(self):
        if self._table is not None:
            return self._table._init_op
        with ops.name_scope(None, 'init'):
            return control_flow_ops.no_op()

    @property
    @deprecated('2018-12-15', 'Use `initializer` instead.')
    def init(self):
        return self.initializer

    @property
    def resource_handle(self):
        if self._table is not None:
            return self._table.resource_handle
        return None

    @property
    def name(self):
        return self._table_name

    def size(self, name=None):
        """Compute the number of elements in this table."""
        with ops.name_scope(name, '%s_Size' % self.name):
            if self._table:
                tsize = self._table.size()
            else:
                tsize = ops.convert_to_tensor(0, dtype=dtypes.int64)
            return tsize + self._num_oov_buckets

    def _get_string_to_hash_bucket_fn(self, hasher_spec):
        """Returns the string_to_hash_bucket op to use based on `hasher_spec`."""
        if not isinstance(hasher_spec, HasherSpec):
            raise TypeError(f'`hasher_spec` must be of type HasherSpec, got {type(hasher_spec)}.')
        if hasher_spec.hasher == 'fasthash':
            return string_ops.string_to_hash_bucket_fast
        if hasher_spec.hasher == 'legacy':
            return string_ops.string_to_hash_bucket
        if hasher_spec.hasher == 'stronghash':
            return functools.partial(string_ops.string_to_hash_bucket_strong, key=hasher_spec.key)
        raise ValueError(f'Found unknown hasher {hasher_spec.hasher} in `hasher_spec`')

    def lookup(self, keys, name=None):
        """Looks up `keys` in the table, outputs the corresponding values.

    It assigns out-of-vocabulary keys to buckets based in their hashes.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: Optional name for the op.

    Returns:
      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,
      otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` doesn't match the table key data type.
    """
        if keys.dtype.base_dtype != self._key_dtype:
            raise TypeError(f'Dtype of argument `keys` must be {self._key_dtype}, received: {keys.dtype}')
        values = keys
        if isinstance(keys, (sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor)):
            values = keys.values
        if self._table and self._table.key_dtype.base_dtype == dtypes.int64:
            values = math_ops.cast(values, dtypes.int64)
        if self._num_oov_buckets == 0:
            ids = self._table.lookup(values, name=name)
        else:
            with ops.name_scope(name, '%s_Lookup' % self.name):
                str_to_hash_bucket = self._get_string_to_hash_bucket_fn(self._hasher_spec)
                buckets = str_to_hash_bucket(_as_string(values), num_buckets=self._num_oov_buckets, name='hash_bucket')
                if self._table:
                    ids = self._table.lookup(values)
                    buckets = math_ops.add(buckets, self._table.size())
                    is_id_non_default = math_ops.not_equal(ids, self._table.default_value)
                    ids = array_ops.where_v2(is_id_non_default, ids, buckets)
                else:
                    ids = buckets
        if isinstance(keys, sparse_tensor.SparseTensor):
            return sparse_tensor.SparseTensor(keys.indices, ids, keys.dense_shape)
        elif isinstance(keys, ragged_tensor.RaggedTensor):
            return keys.with_values(ids)
        return ids