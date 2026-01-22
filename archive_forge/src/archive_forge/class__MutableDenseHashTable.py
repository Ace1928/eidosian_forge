from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.checkpoint import saveable_compat
@saveable_compat.legacy_saveable_name('table')
class _MutableDenseHashTable(lookup_ops.LookupInterface):
    """Copy of tf.contrib.lookup.MutableDenseHashTable."""

    def __init__(self, key_dtype, value_dtype, default_value, empty_key, deleted_key, initial_num_buckets=None, shared_name=None, name='MutableDenseHashTable', checkpoint=True):
        """Creates an empty `_MutableDenseHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      empty_key: the key to use to represent empty buckets internally. Must not
        be used in insert, remove or lookup operations.
      deleted_key: the key to use to represent deleted buckets internally. Must
        not be used in insert, remove or lookup operations and be different from
        the empty_key.
      initial_num_buckets: the initial number of buckets.
      shared_name: If non-empty, this table will be shared under the given name
        across multiple sessions.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.

    Returns:
      A `_MutableDenseHashTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
        self._default_value = ops.convert_to_tensor(default_value, dtype=value_dtype, name='default_value')
        self._key_dtype = key_dtype
        self._value_dtype = value_dtype
        self._initial_num_buckets = initial_num_buckets
        self._value_shape = self._default_value.get_shape()
        self._checkpoint = checkpoint
        self._name = name
        self._empty_key = ops.convert_to_tensor(empty_key, dtype=key_dtype, name='empty_key')
        self._deleted_key = ops.convert_to_tensor(deleted_key, dtype=key_dtype, name='deleted_key')
        if tf.executing_eagerly() and shared_name is None:
            shared_name = 'table_%d' % (ops.uid(),)
        self._shared_name = shared_name
        super(_MutableDenseHashTable, self).__init__(key_dtype, value_dtype)
        self._resource_handle = self._create_resource()
        if checkpoint:
            saveable = _MutableDenseHashTable._Saveable(self, name)
            if not tf.executing_eagerly():
                tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.SAVEABLE_OBJECTS, saveable)

    def _create_resource(self):
        use_node_name_sharing = self._checkpoint and self._shared_name is None
        table_ref = gen_lookup_ops.mutable_dense_hash_table_v2(empty_key=self._empty_key, deleted_key=self._deleted_key, shared_name=self._shared_name, use_node_name_sharing=use_node_name_sharing, value_dtype=self._value_dtype, value_shape=self._value_shape, initial_num_buckets=self._initial_num_buckets, name=self._name)
        if tf.executing_eagerly():
            self._table_name = None
        else:
            self._table_name = table_ref.op.name.split('/')[-1]
        return table_ref

    @property
    def name(self):
        return self._table_name

    def size(self, name=None):
        """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
        with ops.name_scope(name, '%s_Size' % self.name, [self.resource_handle]) as name:
            with ops.colocate_with(self.resource_handle):
                return gen_lookup_ops.lookup_table_size_v2(self.resource_handle, name=name)

    def lookup(self, keys, name=None):
        """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
        with ops.name_scope(name, '%s_lookup_table_find' % self.name, [self.resource_handle, keys]) as name:
            keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name='keys')
            with ops.colocate_with(self.resource_handle):
                values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, keys, self._default_value, name=name)
        return values

    def insert(self, keys, values, name=None):
        """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
        with ops.name_scope(name, '%s_lookup_table_insert' % self.name, [self.resource_handle, keys, values]) as name:
            keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name='keys')
            values = ops.convert_to_tensor(values, dtype=self._value_dtype, name='values')
            with ops.colocate_with(self.resource_handle):
                op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys, values, name=name)
            return op

    def export(self, name=None):
        """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
        with ops.name_scope(name, '%s_lookup_table_export_values' % self.name, [self.resource_handle]) as name:
            with ops.colocate_with(self.resource_handle):
                exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(self.resource_handle, self._key_dtype, self._value_dtype, name=name)
        return (exported_keys, exported_values)

    def _serialize_to_tensors(self):
        tesnors = self.export()
        return {'-keys': tesnors[0], '-values': tesnors[1]}

    def _restore_from_tensors(self, restored_tensors):
        with ops.colocate_with(self.resource_handle):
            return gen_lookup_ops.lookup_table_import_v2(self.resource_handle, restored_tensors['-keys'], restored_tensors['-values'])

    class _Saveable(BaseSaverBuilder.SaveableObject):
        """SaveableObject implementation for _MutableDenseHashTable."""

        def __init__(self, table, name):
            tensors = table.export()
            specs = [BaseSaverBuilder.SaveSpec(tensors[0], '', name + '-keys'), BaseSaverBuilder.SaveSpec(tensors[1], '', name + '-values')]
            super(_MutableDenseHashTable._Saveable, self).__init__(table, specs, name)

        def restore(self, restored_tensors, restored_shapes):
            del restored_shapes
            with ops.colocate_with(self.op.resource_handle):
                return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle, restored_tensors[0], restored_tensors[1])