import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.StructuredTensor')
class StructuredTensor(extension_type.BatchableExtensionType):
    """A multidimensional collection of structures with the same schema.

  A **`StructuredTensor`** is a multi-dimensional collection of ***structures***
  with the same ***schema***, where:

  * A ***schema*** is a collection of fields, each of which has a name and type.
  * A ***structure*** maps each field in the schema to a tensor value (which
    could be a nested StructuredTensor).

  As an important special case, a 1D `StructuredTensor` encodes a 2D table,
  where columns are heterogeneous `Tensor`s, and rows are the aligned elements
  in each of those `Tensor`s.

  Internally, StructuredTensors use a "field-major" encoding: for each leaf
  field, there is a single tensor that stores the value of that field for all
  structures in the `StructuredTensor`.

  ### Examples

  >>> # A scalar StructuredTensor describing a single person.
  >>> s1 = tf.experimental.StructuredTensor.from_pyval(
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]})
  >>> s1.shape
  TensorShape([])
  >>> s1["age"]
  <tf.Tensor: shape=(), dtype=int32, numpy=82>

  >>> # A vector StructuredTensor describing three people.
  >>> s2 = tf.experimental.StructuredTensor.from_pyval([
  ...     {"age": 12, "nicknames": ["Josaphine"]},
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]},
  ...     {"age": 42, "nicknames": ["Elmo"]}])
  >>> s2.shape
  TensorShape([3])
  >>> s2[0]["age"]
  <tf.Tensor: shape=(), dtype=int32, numpy=12>


  ### Field Paths

  A *field path* is a tuple of field names, specifying the path to a nested
  field.
  """
    _fields: Mapping[str, _FieldValue]
    _ragged_shape: dynamic_ragged_shape.DynamicRaggedShape
    __name__ = 'tf.StructuredTensor'
    FieldName = Union[str, Sequence[str]]

    def __init__(self, fields: Mapping[str, _FieldValue], ragged_shape: dynamic_ragged_shape.DynamicRaggedShape):
        self._fields = fields
        self._ragged_shape = ragged_shape

    @classmethod
    def _old_init(cls, fields, shape, nrows, row_partitions, internal=False):
        """Private constructor -- use factory methods to create StructuredTensors.

    This constructor builds a `StructuredTensor` from the given attributes,
    performing minimal validation.

    Args:
      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or
        `StructuredTensor`.  (This dict is not copied, so the caller must ensure
        that it does not get mutated via leaked references.)
      shape: `tf.TensorShape` with statically known rank.
      nrows: scalar integer `tf.Tensor`, or `None` if `shape.rank==0`.
      row_partitions: tuple of `RowPartition`s, with length `shape.rank-1`.
      internal: ignored argument.

    Returns:
      a StructuredTensor.
    """
        assert isinstance(fields, dict), fields
        assert isinstance(shape, tensor_shape.TensorShape), shape
        assert nrows is None or isinstance(nrows, tensor.Tensor), nrows
        assert row_partitions is None or isinstance(row_partitions, tuple), row_partitions
        return StructuredTensor(fields=fields, ragged_shape=_dynamic_ragged_shape_init(fields, shape, nrows, row_partitions))

    @classmethod
    def from_shape(cls, ragged_shape: dynamic_ragged_shape.DynamicRaggedShape) -> 'StructuredTensor':
        """Creates a `StructuredTensor` with no fields and ragged_shape.

    Args:
      ragged_shape: the shape of the structured tensor.

    Returns:
      a StructuredTensor with no fields and ragged_shape.
    """
        return StructuredTensor(fields={}, ragged_shape=ragged_shape)

    @classmethod
    def from_fields(cls, fields, shape=(), nrows=None, row_partitions=None, validate=False):
        """Creates a `StructuredTensor` from a dictionary of fields.

    Args:
      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or
        `StructuredTensor`, providing the values for individual fields in each
        structure.  If `shape.rank > 0`, then every tensor in `fields` must have
        the same shape in the first `shape.rank` dimensions; and that shape must
        be compatible with `shape`; and `result[i1...iN][key] =
        fields[key][i1...iN]` (where `N==shape.rank`).
      shape: A `TensorShape`: static information about the shape of the
        `StructuredTensor`.  Must have a known `rank`.  Defaults to scalar shape
        (i.e. `rank=0`).
      nrows: scalar integer tensor containing the number of rows in this
        `StructuredTensor`.  Should only be specified if `shape.rank > 0`.
        Default value is inferred from the `fields` values.  If `fields` is
        empty, then this must be specified.
      row_partitions: A list of `RowPartition`s describing the (possibly ragged)
        shape of this `StructuredTensor`.  Should only be specified if
        `shape.rank > 1`.  Default value is inferred from the `fields` values.
        If `fields` is empty, then this must be specified.
      validate: If true, then add runtime validation ops that check that the
        field values all have compatible shapes in the outer `shape.rank`
        dimensions.

    Returns:
      A `StructuredTensor`.

    Examples:

      >>> tf.experimental.StructuredTensor.from_fields({'x': 1, 'y': [1, 2, 3]})
      <StructuredTensor(
        fields={
          "x": tf.Tensor(1, shape=(), dtype=int32),
          "y": tf.Tensor([1 2 3], shape=(3,), dtype=int32)},
        shape=())>

      >>> tf.experimental.StructuredTensor.from_fields(
      ...     {'foo': [1, 2], 'bar': [3, 4]}, shape=[2])
      <StructuredTensor(
        fields={
          "bar": tf.Tensor([3 4], shape=(2,), dtype=int32),
          "foo": tf.Tensor([1 2], shape=(2,), dtype=int32)},
        shape=(2,))>
    """
        shape = tensor_shape.as_shape(shape)
        rank = shape.rank
        if rank is None:
            raise ValueError("StructuredTensor's shape must have known rank.")
        if not isinstance(fields, dict):
            raise TypeError('fields must be a dictionary, got %s' % type(fields).__name__)
        if rank < 2 and row_partitions:
            raise ValueError('row_partitions must be None or [] if shape.rank<2')
        if rank == 0 and nrows is not None:
            raise ValueError('nrows must be None if shape.rank==0')
        if row_partitions is not None:
            row_partitions = tuple(row_partitions)
            if len(row_partitions) != max(0, rank - 1):
                raise ValueError('len(row_partitions) must be shape.rank-1')
        elif rank < 2:
            row_partitions = ()
        fields = dict(fields)
        with ops.name_scope(None, 'StructuredTensor', fields.values()):
            shape = _dynamic_ragged_shape_init(fields, shape, nrows, row_partitions)
            if shape.rank > 1:
                shape = shape._with_num_row_partitions(shape.rank - 1)
            for key, value in fields.items():
                if not isinstance(key, str):
                    raise TypeError(f'Unexpected type for key in `fields`: {key}')
                if not _FIELD_NAME_RE.match(key):
                    raise ValueError('Field name %r is not currently allowed.' % key)
                fields[key] = _convert_to_structured_field_value(value)
                fields = dict([(k, _replace_row_partitions(v, row_partitions)) for k, v in fields.items()])
            return cls(fields=fields, ragged_shape=shape)

    @classmethod
    def from_fields_and_rank(cls, fields: Mapping[str, _FieldValue], rank: int, validate: bool=False, dtype: Optional[dtypes.DType]=None) -> 'StructuredTensor':
        """Creates a `StructuredTensor` from a nonempty dictionary of fields.

    Note that if the shape dtype is not specified, the shape dtype will be
    inferred from any fields that have a shape dtype. If fields differ, then
    int64 will be preferred to int32, because coercing from int32 to int64 is
    safer than coercing from int64 to int32.

    If there are no ragged fields, then it will be int64 by default, but this
    will be changed to int32 in the future.

    Args:
      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or
        `StructuredTensor`, providing the values for individual fields in each
        structure.  If `rank > 0`, then every tensor in `fields` must have the
        same shape in the first `rank` dimensions. Cannot be empty.
      rank: The rank of the resulting structured tensor.
      validate: If true, then add runtime validation ops that check that the
        field values all have compatible shapes in the outer `rank` dimensions.
      dtype: If specified, then forces dtype of the shape to be this.

    Returns:
      A `StructuredTensor`.
    Examples:
      >>> tf.experimental.StructuredTensor.from_fields_and_rank(
      ...     {'x': 1, 'y': [1, 2, 3]}, 0)
      <StructuredTensor(
        fields={
          "x": tf.Tensor(1, shape=(), dtype=int32),
          "y": tf.Tensor([1 2 3], shape=(3,), dtype=int32)},
        shape=())>
      >>> StructuredTensor.from_fields_and_rank({'foo': [1, 2], 'bar': [3, 4]},
      ...                              1)
      <StructuredTensor(
        fields={
          "bar": tf.Tensor([3 4], shape=(2,), dtype=int32),
          "foo": tf.Tensor([1 2], shape=(2,), dtype=int32)},
        shape=(2,))>
    """
        if not fields:
            raise ValueError('Must provide at least one field')
        if not isinstance(rank, int):
            raise ValueError('rank must be an integer')
        if rank < 0:
            raise ValueError('rank must be nonnegative')
        fields = {k: _convert_to_structured_field_value(v) for k, v in fields.items()}
        if dtype is None:
            dtype = _find_shape_dtype(fields, None, None)
        fields = _fields_with_dtype(fields, dtype)
        shape = _shape_from_fields(fields, rank, dtype)
        if rank > 1:
            shape = shape._with_num_row_partitions(rank - 1)
        new_rp = shape._row_partitions
        fields = {k: _replace_row_partitions(v, new_rp) for k, v in fields.items()}
        return StructuredTensor(fields=fields, ragged_shape=shape)

    def with_updates(self, updates: Dict[FieldName, Union[_FieldValue, _FieldFn, None]], validate: bool=False) -> 'StructuredTensor':
        """Creates a new `StructuredTensor` with the updated fields.

    If this `StructuredTensor` is a scalar, and `k` is the `FieldName` being
    updated and `v` the new value, then:

    ```
    result[k] = v              # If (k, v) is in updates and v is a FieldValue
    result[k] = f(self[k])     # If (k, f) is in updates and f is a FieldFn
    result[k] = self[k]        # If k is in self.field_names but not in updates
    ```

    If this `StructuredTensor` has rank `N` and shape `[D1...DN]`, then each
    FieldValue `v` in `updates` must have shape `[D1...DN, ...]`, that is,
    prefixed with the same shape as the `StructuredTensor`. Then the resulting
    `StructuredTensor` will have:

    ```
    result[i1...iN][k] = v[i1...iN]                        # (k, v) in updates
    result[i1...iN][k] = f(self.field_value(k))[i1...iN]   # (k, f) in updates
    result[i1...iN][k] = self[i1...iN][k]                  # k not in updates
    ```

    Note that `result.shape` is always equal to `self.shape` (but the shapes
    of nested StructuredTensors may be changed if they are updated with new
    values).

    Args:
      updates: A dictionary mapping `FieldName` to either a `FieldValue` to be
        used to update, or a `FieldFn` that will transform the value for the
        given `FieldName`. `FieldName` can be a string for a direct field, or a
        sequence of strings to refer to a nested sub-field. `FieldFn` is a
        function that takes a `FieldValue` as input and should return a
        `FieldValue`. All other fields are copied over to the new
        `StructuredTensor`. New `FieldName` can be given (to add new fields),
        but only to existing `StructuredTensor`, it won't automatically create
        new nested structures -- but one can create a whole `StructureTensor`
        sub-structure and set that into an existing structure. If the new value
        is set to `None`, it is removed.
      validate: If true, then add runtime validation ops that check that the
        field values all have compatible shapes in the outer `shape.rank`
        dimensions.

    Returns:
      A `StructuredTensor`.

    Raises:
      `ValueError`: If the any of the `FieldName` keys points to non-existent
        sub-structures, if parent and child nodes are updated, if shapes
        change, if a delete update is given for a non-existent field, or if a
        `FieldFn` transforming function is given for a `FieldName` that doesn't
        yet exist.

    Examples:

    >>> shoes_us = tf.experimental.StructuredTensor.from_pyval([
    ...    {"age": 12, "nicknames": ["Josaphine"],
    ...       "shoes": {"sizes": [8.0, 7.5, 7.5]}},
    ...    {"age": 82, "nicknames": ["Bob", "Bobby"],
    ...        "shoes": {"sizes": [11.0, 11.5, 12.0]}},
    ...    {"age": 42, "nicknames": ["Elmo"],
    ...        "shoes": {"sizes": [9.0, 9.5, 10.0]}}])
    >>> def us_to_europe(t):
    ...   return tf.round(t * 2.54 + 17.0)  # Rough approximation.
    >>> shoe_sizes_key = ("shoes", "sizes")
    >>> shoes_eu = shoes_us.with_updates({shoe_sizes_key: us_to_europe})
    >>> shoes_eu.field_value(shoe_sizes_key)
    <tf.RaggedTensor [[37.0, 36.0, 36.0], [45.0, 46.0, 47.0],
    [40.0, 41.0, 42.0]]>
    """
        updates_items = [(_normalize_field_name_to_tuple(name), value) for name, value in updates.items()]
        updates_items = sorted(updates_items)
        for i in range(1, len(updates_items)):
            name = updates_items[i][0]
            prev_name = updates_items[i - 1][0]
            if name[:len(prev_name)] == prev_name:
                raise ValueError('`StructuredTensor.with_updates` does not allow both parent and child nodes to be updated: parent={}, child={}. If needed you can update child nodes in the parent update value.'.format(prev_name, name))
        return self._with_updates_impl((), updates_items, validate)

    def _with_updates_impl(self, error_prefix: Tuple[str, ...], updates: List[Tuple[FieldName, Union[_FieldValue, _FieldFn]]], validate: bool) -> 'StructuredTensor':
        """Recursive part of `with_updates` implementation."""
        new_fields = dict(self._fields)

        def name_fullpath(name: Sequence[str]) -> str:
            return str(error_prefix + (name,))

        def apply_value(name: str, value: Union[_FieldValue, _FieldFn]) -> _FieldValue:
            if callable(value):
                if name not in new_fields:
                    raise ValueError('`StructuredTensor.with_updates` cannot update the field {} because a transforming function was given, but that field does not already exist.'.format(name_fullpath(name)))
                value = value(new_fields[name])
            return value
        for name, value in updates:
            if not name or not name[0]:
                raise ValueError('`StructuredTensor.with_updates` does not allow empty names {}.'.format(name_fullpath(name)))
            if len(name) == 1:
                name = name[0]
                if value is None:
                    if name not in new_fields:
                        raise ValueError('`StructuredTensor.with_updates` cannot delete field {} because it is not present.'.format(name_fullpath(name)))
                    new_fields.pop(name)
                else:
                    new_fields[name] = apply_value(name, value)
            else:
                prefix = name[0]
                suffix = name[1:]
                if prefix not in new_fields:
                    raise ValueError('`StructuredTensor.with_updates` cannot create new sub-field {} if parent field {} is not set.'.format(error_prefix + tuple(name), name_fullpath(prefix)))
                current_value = new_fields[prefix]
                if not isinstance(current_value, StructuredTensor):
                    raise ValueError('`StructuredTensor.with_updates` cannot create new sub-field {} if parent structure {} is not a `StructuredTensor` that can contain sub-structures -- it is a `{}`.'.format(error_prefix + tuple(name), name_fullpath(prefix), type(current_value)))
                one_update = [(suffix, value)]
                value = current_value._with_updates_impl(error_prefix + (prefix,), one_update, validate)
                new_fields[prefix] = value
        try:
            return StructuredTensor.from_fields(new_fields, shape=self.shape, row_partitions=self.row_partitions, nrows=self.nrows(), validate=validate)
        except ValueError as e:
            msg = '`StructuredTensor.with_updates` failed'
            if error_prefix:
                msg = '{} for field {}'.format(msg, error_prefix)
            raise ValueError(msg) from e

    def _promote_helper(self, source_path, new_parent_path):
        """Creates a promoted field without adding it to the structure.

    Args:
      source_path: the source path in the structured tensor.
      new_parent_path: the new parent path. Must be a prefix of source_path.

    Returns:
      a composite tensor of source_path promoted.
    Raises:
      ValueError: if the shape of the field is unknown and the right strategy
      cannot be determined.
    """
        current_field = self.field_value(source_path)
        new_parent_rank = self.field_value(new_parent_path).rank
        parent_rank = self.field_value(source_path[:-1]).rank
        if new_parent_rank == parent_rank:
            return current_field
        current_field_rank = current_field.shape.rank
        if current_field_rank is None:
            raise ValueError('Cannot determine if dimensions should be merged.')
        inner_dim = min(parent_rank, current_field_rank - 1)
        if inner_dim <= new_parent_rank:
            return current_field
        return _merge_dims_generic(current_field, new_parent_rank, inner_dim)

    def promote(self, source_path, new_name):
        """Promotes a field, merging dimensions between grandparent and parent.

    >>> d = [
    ...  {'docs': [{'tokens':[1, 2]}, {'tokens':[3]}]},
    ...  {'docs': [{'tokens':[7]}]}]
    >>> st = tf.experimental.StructuredTensor.from_pyval(d)
    >>> st2 =st.promote(('docs','tokens'), 'docs_tokens')
    >>> st2[0]['docs_tokens']
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>
    >>> st2[1]['docs_tokens']
    <tf.Tensor: shape=(1,), dtype=int32, numpy=array([7], dtype=int32)>

    Args:
      source_path: the path of the field or substructure to promote; must have
        length at least 2.
      new_name: the name of the new field (must be a string).

    Returns:
      a modified structured tensor with the new field as a child of the
      grandparent of the source_path.

    Raises:
      ValueError: if source_path is not a list or a tuple or has a length
        less than two, or new_name is not a string, or the rank
        of source_path is unknown and it is needed.
    """
        if not isinstance(new_name, str):
            raise ValueError('new_name is not a string')
        if not isinstance(source_path, (list, tuple)):
            raise ValueError('source_path must be a list or tuple')
        if len(source_path) < 2:
            raise ValueError('source_path must have length at least two')
        grandparent_path = source_path[:-2]
        new_field = self._promote_helper(source_path, grandparent_path)
        new_path = grandparent_path + (new_name,)
        return self.with_updates({new_path: new_field})

    @property
    def rank(self):
        """The rank of this StructuredTensor.  Guaranteed not to be `None`."""
        return self._ragged_shape.rank

    @property
    def shape(self):
        """The static shape of this StructuredTensor.

    The returned `TensorShape` is guaranteed to have a known rank, but the
    individual dimension sizes may be unknown.

    Returns:
      `tf.TensorShape`
    """
        return self._ragged_shape._to_tensor_shape()

    @property
    def _row_partitions(self):
        """Deprecated form of row_partitions."""
        return self.row_partitions

    @property
    def row_partitions(self):
        """A tuple of `RowPartition`s defining the shape of this `StructuredTensor`.

    When `self.rank <= 1`, this tuple will be empty.

    When `self.rank > 1`, these `RowPartitions` define the shape of the
    `StructuredTensor` by describing how a flat (1D) list of structures can be
    repeatedly partitioned to form a higher-dimensional object.  In particular,
    the flat list is first partitioned into sublists using `row_partitions[-1]`,
    and then those sublists are further partitioned using `row_partitions[-2]`,
    etc.  The following examples show the row partitions used to describe
    several different `StructuredTensor`, each of which contains 8 copies of
    the same structure (`x`):

    >>> x = {'a': 1, 'b': ['foo', 'bar', 'baz']}       # shape = [] (scalar)

    >>> s1 = [[x, x, x, x], [x, x, x, x]]              # shape = [2, 4]
    >>> tf.experimental.StructuredTensor.from_pyval(s1).row_partitions
    (tf.RowPartition(row_splits=[0 4 8]),)

    >>> s2 = [[x, x], [x, x], [x, x], [x, x]]          # shape = [4, 2]
    >>> tf.experimental.StructuredTensor.from_pyval(s2).row_partitions
    (tf.RowPartition(row_splits=[0 2 4 6 8]),)

    >>> s3 = [[x, x, x], [], [x, x, x, x], [x]]        # shape = [2, None]
    >>> tf.experimental.StructuredTensor.from_pyval(s3).row_partitions
    (tf.RowPartition(row_splits=[0 3 3 7 8]),)

    >>> s4 = [[[x, x], [x, x]], [[x, x], [x, x]]]      # shape = [2, 2, 2]
    >>> tf.experimental.StructuredTensor.from_pyval(s4).row_partitions
    (tf.RowPartition(row_splits=[0 2 4]),
     tf.RowPartition(row_splits=[0 2 4 6 8]))


    >>> s5 = [[[x, x], [x]], [[x, x]], [[x, x], [x]]]  # shape = [3, None, None]
    >>> tf.experimental.StructuredTensor.from_pyval(s5).row_partitions
    (tf.RowPartition(row_splits=[0 2 3 5]),
     tf.RowPartition(row_splits=[0 2 3 5 7 8]))

    Note that shapes for nested fields (such as `x['b']` in the above example)
    are not considered part of the shape of a `StructuredTensor`, and are not
    included in `row_partitions`.

    If this `StructuredTensor` has a ragged shape (i.e., if any of the
    `row_partitions` is not uniform in size), then all fields will be encoded
    as either `RaggedTensor`s or `StructuredTensor`s with these `RowPartition`s
    used to define their outermost `self.rank` dimensions.

    Returns:
      A `tuple` of `RowPartition` objects with length `self.rank - 1`
      (or `0` if `self.rank < 2`)

    """
        if self.rank < 2:
            return ()
        return self._ragged_shape._as_row_partitions()

    def nrows(self):
        """The number of rows in this StructuredTensor (if rank>0).

    This means the length of the outer-most dimension of the StructuredTensor.

    Notice that if `self.rank > 1`, then this equals the number of rows
    of the first row partition. That is,
    `self.nrows() == self.row_partitions[0].nrows()`.

    Otherwise `self.nrows()` will be the first dimension of the field values.

    Returns:
      A scalar integer `Tensor` (or `None` if `self.rank == 0`).
    """
        if self.rank == 0:
            return None
        return self._ragged_shape[0]

    def with_shape_dtype(self, dtype: dtypes.DType) -> 'StructuredTensor':
        if dtype == self._ragged_shape.dtype:
            return self
        return StructuredTensor(fields=_fields_with_dtype(self._fields, dtype), ragged_shape=self._ragged_shape.with_dtype(dtype))

    def _is_eager(self):
        """True if all fields are composed of eager tensors."""
        tensors = nest.flatten(self, expand_composites=True)
        return all((isinstance(t, ops.EagerTensor) for t in tensors))

    def field_names(self):
        """Returns the string field names for this `StructuredTensor`."""
        return tuple(self._fields.keys())

    def field_value(self, field_name):
        """Returns the tensor value for the specified field or path.

    If `field_name` is a `string`, then it names a field directly owned by this
    `StructuredTensor`.  If this `StructuredTensor` has shape `[D1...DN]`, then
    the returned tensor will have shape `[D1...DN, V1...VM]`, where the slice
    `result[d1...dN]` contains the field value for the structure at
    `self[d1...dN]`.

    If `field_name` is a `tuple` of `string`, then it specifies a path to a
    field owned by nested `StructuredTensor`.  In particular,
    `struct.field_value((f1, f2, ..., fN))` is equivalent to
    `struct.field_value(f1).field_value(f2)....field_value(fN)`

    Args:
      field_name: `string` or `tuple` of `string`: The field whose values should
        be returned.

    Returns:
      `Tensor`, `StructuredTensor`, or `RaggedTensor`.

    Raises:
      KeyError: If the given field_name is not found.
    """
        if isinstance(field_name, (list, tuple)):
            value = self
            for f in field_name:
                if not isinstance(value, StructuredTensor):
                    raise KeyError('Field path {} not found in {}'.format(field_name, self))
                value = value.field_value(f)
            return value
        return self._fields[field_name]

    def __getitem__(self, key):
        """Returns the specified piece of this StructuredTensor.

    * If `struct_tensor` is scalar (i.e., a single structure), then
      `struct_tensor[f]` returns the value of field `f` (where `f` must be a
      string).

    * If `struct_tensor` is non-scalar (i.e., a vector or higher-dimensional
      tensor of structures), `struct_tensor[i]` selects an element or slice of
      the tensor using standard Python semantics (e.g., negative values index
      from the end).  `i` may have any of the following types:

      * `int` constant
      * `string` constant
      * scalar integer `Tensor`
      * `slice` containing integer constants and/or scalar integer
        `Tensor`s

    #### Multidimensional indexing

    `StructuredTensor` supports multidimensional indexing.  I.e., `key` may be a
    `tuple` of values, indexing or slicing multiple dimensions at once.  For
    example, if `people` is a vector of structures, each of which has a vector-
    valued `names` field, then `people[3, 'names', 0]` is equivalent to
    `people[3]['names'][0]`; and `people[:, 'names', :]` will return a (possibly
    ragged) matrix of names, with shape `[num_people, num_names_per_person]`.

    Args:
      key: Indicates which piece of the StructuredTensor to return.

    Returns:
      A `Tensor`, `StructuredTensor`, or `RaggedTensor`.
    """
        if isinstance(key, list):
            key = tuple(key)
        elif not isinstance(key, tuple):
            key = (key,)
        if not key:
            return self
        if self.rank == 0:
            return self._scalar_getitem(key)
        else:
            return self._tensor_getitem(key)

    def _scalar_getitem(self, key):
        if isinstance(key[0], slice) and key[0].start is None and (key[0].stop is None) and (key[0].step is None):
            fields = dict(((field_name, field_value.__getitem__(key[1:])) for field_name, field_value in self._fields.items()))
            return StructuredTensor.from_fields(fields, self.shape)
        elif not isinstance(key[0], compat.bytes_or_text_types):
            raise ValueError("Key for indexing a StructuredTensor must be a string or a full slice (':')")
        return self._fields[key[0]].__getitem__(key[1:])

    def _tensor_getitem(self, key):
        rank = self.rank
        if len(key) <= rank:
            new_fields = dict(((field_name, field_value.__getitem__(key)) for field_name, field_value in self._fields.items()))
            result_shape = self.shape.as_list()
            for d, k in enumerate(key):
                if isinstance(k, slice):
                    if not (k.start is None and k.stop is None and (k.step is None)):
                        result_shape[d] = None
                elif isinstance(k, (int, tensor.Tensor)):
                    result_shape[d] = -1
                elif k is None:
                    raise ValueError('Slicing not supported for tf.newaxis')
                else:
                    raise ValueError('Slicing not supported for %r' % k)
            result_shape = [d for d in result_shape if d != -1]
            return StructuredTensor.from_fields(new_fields, result_shape)
        else:
            if not isinstance(key[rank], compat.bytes_or_text_types):
                raise ValueError('Key for indexing a StructuredTensor must be a string')
            return self._fields[key[rank]].__getitem__(key[:rank] + key[rank + 1:])

    def __repr__(self):
        fields = sorted(self._fields.items())
        fields = ((k, str(v).replace('\n', '\n            ')) for k, v in fields)
        fields = ('"{}": {}'.format(k, v) for k, v in fields)
        dict_repr = ',\n        '.join(fields)
        return '<StructuredTensor(\n    fields={\n        %s},\n    shape=%s)>' % (dict_repr, self.shape)

    def to_pyval(self):
        """Returns this StructuredTensor as a nested Python dict or list of dicts.

    Converts this `StructuredTensor` to a nested python value:

    * `StructTensors` with `rank=0` are converted into a dictionary, with an
      entry for each field.  Field names are used as keys and field values are
      converted to python values.  In particular:

      * Scalar Tensor fields are converted to simple values (such as
        `int` or `float` or `string`)
      * Non-scalar Tensor fields and RaggedTensor fields are converted to
        nested lists of simple values.
      * StructuredTensor fields are converted recursively using `to_pyval`.

    * `StructTensors` with `rank>0` are converted to nested python `list`s,
      containing one dictionary for each structure (where each structure's
      dictionary is defined as described above).

    Requires that all fields are Eager tensors.

    >>> tf.experimental.StructuredTensor.from_fields(
    ...     {'a': [1, 2, 3]}, [3]).to_pyval()
    [{'a': 1}, {'a': 2}, {'a': 3}]

    Note that `StructuredTensor.from_pyval(pyval).to_pyval() == pyval`.

    Returns:
      A nested Python dict or list of dicts.
    """
        if not self._is_eager():
            raise ValueError('StructuredTensor.to_pyval() is only supported in eager mode.')
        result = {}
        for key, value in self._fields.items():
            if isinstance(value, ops.EagerTensor):
                value = value.numpy()
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, ragged_tensor.RaggedTensor):
                value = value.to_list()
            elif isinstance(value, StructuredTensor):
                value = value.to_pyval()
            result[key] = value
        if len(self.shape) > 0:
            if not result:
                return _empty_dict_pylist_from_row_partitions(self.row_partitions, self.nrows())
            return _pyval_field_major_to_node_major(list(result.keys()), list(result.values()), self.rank)
        else:
            return result

    @classmethod
    def from_pyval(cls, pyval, typespec=None):
        """Constructs a StructuredTensor from a nested Python structure.

    >>> tf.experimental.StructuredTensor.from_pyval(
    ...     {'a': [1, 2, 3], 'b': [[4, 5], [6, 7]]})
    <StructuredTensor(
        fields={
          "a": tf.Tensor([1 2 3], shape=(3,), dtype=int32),
          "b": <tf.RaggedTensor [[4, 5], [6, 7]]>},
        shape=())>

    Note that `StructuredTensor.from_pyval(pyval).to_pyval() == pyval`.

    Args:
      pyval: The nested Python structure that should be used to create the new
        `StructuredTensor`.
      typespec: A `StructuredTensor.Spec` specifying the expected type for each
        field. If not specified, then all nested dictionaries are turned into
        StructuredTensors, and all nested lists are turned into Tensors (if
        rank<2) or RaggedTensors (if rank>=2).

    Returns:
      A `StructuredTensor`.
    """
        return cls._from_pyval(pyval, typespec, ())

    @classmethod
    def _from_pyval(cls, pyval, typespec, path_so_far):
        """Helper function for from_pyval.


    Args:
      pyval: The nested Python structure that should be used to create the new
        `StructuredTensor`.
      typespec: A `StructuredTensor.Spec` specifying the expected type for each
        field. If not specified, then all nested dictionaries are turned into
        StructuredTensors, and all nested lists are turned into Tensors (if
        rank<2) or RaggedTensors (if rank>=2).
      path_so_far: the path of fields that led here (for error messages).

    Returns:
      A `StructuredTensor`.
    """
        if isinstance(pyval, dict):
            return cls._from_pydict(pyval, typespec, path_so_far)
        elif isinstance(pyval, (list, tuple)):
            keys = set()
            rank = _pyval_find_struct_keys_and_depth(pyval, keys)
            if rank is not None:
                return cls._from_pylist_of_dict(pyval, keys, rank, typespec, path_so_far)
            else:
                return cls._from_pylist_of_value(pyval, typespec, path_so_far)
        else:
            return cls._from_pyscalar(pyval, typespec, path_so_far)

    @classmethod
    def _from_pydict(cls, pyval, typespec, path_so_far):
        """Converts python dictionary `pyval` to a StructuredTensor with rank=0."""
        if typespec is None:
            fields = dict(((k, cls._from_pyval(v, None, path_so_far + (k,))) for k, v in pyval.items()))
        else:
            spec_shape = typespec._shape
            field_specs = typespec._field_specs
            if not (isinstance(typespec, StructuredTensor.Spec) and spec_shape.rank == 0 and (set(pyval) == set(field_specs))):
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, pyval, typespec))
            fields = dict(((k, cls._from_pyval(v, field_specs[k], path_so_far + (k,))) for k, v in pyval.items()))
        return StructuredTensor.from_fields(fields=fields, shape=(), validate=False)

    @classmethod
    def _from_pylist_of_dict(cls, pyval, keys, rank, typespec, path_so_far):
        """Converts python list `pyval` to a StructuredTensor with rank>1."""
        fields = dict(((key, []) for key in keys))
        for child in pyval:
            _pyval_update_fields(child, fields, 1)
        if typespec is None:
            shape = tensor_shape.TensorShape([None] * rank)
            for key, target in fields.items():
                fields[key] = cls._from_pyval(target, None, path_so_far + (key,))
        else:
            field_specs = typespec._fields
            if not isinstance(typespec, StructuredTensor.Spec) or set(fields) - set(field_specs):
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, pyval, typespec))
            shape = typespec._shape
            if shape.rank < rank:
                raise ValueError('Value at %r does not match typespec (rank mismatch): %r vs %r' % (path_so_far, pyval, typespec))
            for key, spec in field_specs.items():
                fields[key] = cls._from_pyval(fields.get(key, []), spec, path_so_far + (key,))
        try:
            if not fields and typespec is None:
                return StructuredTensor._from_pylist_of_empty_dict(pyval, rank)
            return StructuredTensor.from_fields(fields=fields, shape=shape, validate=False)
        except Exception as exc:
            raise ValueError('Error parsing path %r' % (path_so_far,)) from exc

    @classmethod
    def _from_pylist_of_empty_dict(cls, pyval, rank):
        """Converts a pylist of empty dictionaries to StructuredTensors."""
        if rank == 0:
            return StructuredTensor.from_fields(fields={}, shape=(), validate=False)
        elif rank == 1:
            nrows = len(pyval)
            shape = (nrows,)
            return StructuredTensor.from_fields(fields={}, shape=shape, nrows=nrows)
        elif rank > 1:
            ragged_zeros = ragged_factory_ops.constant(_dicts_to_zeros(pyval))
            nrows = len(pyval)
            shape = tensor_shape.TensorShape([len(pyval)] + [None] * (rank - 1))
            return StructuredTensor.from_fields(fields={}, shape=shape, row_partitions=ragged_zeros._nested_row_partitions, nrows=nrows)

    @classmethod
    def _from_pylist_of_value(cls, pyval, typespec, path_so_far):
        """Converts python list `pyval` to a Tensor or RaggedTensor with rank>1."""
        if typespec is None:
            try:
                return ragged_factory_ops.constant(pyval)
            except Exception as exc:
                raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
        elif isinstance(typespec, tensor.TensorSpec):
            try:
                result = constant_op.constant(pyval, typespec.dtype)
            except Exception as exc:
                raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
            if not typespec.shape.is_compatible_with(result.shape):
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))
            return result
        elif isinstance(typespec, ragged_tensor.RaggedTensorSpec):
            try:
                return ragged_factory_ops.constant(pyval, dtype=typespec._dtype, ragged_rank=typespec._ragged_rank, row_splits_dtype=typespec._row_splits_dtype, inner_shape=typespec._shape[typespec._ragged_rank + 1:])
            except Exception as exc:
                raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
        elif isinstance(typespec, StructuredTensor.Spec):
            empty_rank = _pyval_empty_list_depth(pyval)
            if empty_rank is None:
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))
            else:
                return cls._from_pylist_of_dict(pyval, set(), empty_rank, typespec, path_so_far)
        else:
            raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))

    @classmethod
    def _from_pyscalar(cls, pyval, typespec, path_so_far):
        """Converts python scalar value `pyval` to a Tensor."""
        if typespec is None:
            try:
                return constant_op.constant(pyval)
            except Exception as exc:
                raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
        else:
            if not (isinstance(typespec, tensor.TensorSpec) and typespec.shape.rank == 0):
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))
            return constant_op.constant(pyval, typespec.dtype)

    def partition_outer_dimension(self, row_partition):
        """Partitions the outer dimension of this StructuredTensor.

    Returns a new `StructuredTensor` with the same values as `self`, where
    the outer dimension is partitioned into two (possibly ragged) dimensions.
    Requires that this StructuredTensor have an outer dimension (i.e.,
    `self.shape.rank > 0`).

    >>> st = tf.experimental.StructuredTensor.from_pyval(
    ...     [{'foo': 12}, {'foo': 33}, {'foo': 99}])
    >>> partition = RowPartition.from_row_lengths([2, 0, 1])
    >>> st.partition_outer_dimension(partition)
    <StructuredTensor(
      fields={
        "foo": <tf.RaggedTensor [[12, 33], [], [99]]>},
      shape=(3, None))>

    Args:
      row_partition: A `RowPartition`.

    Returns:
      A `StructuredTensor` with rank `values.rank + 1`.
    """
        if not isinstance(row_partition, RowPartition):
            raise TypeError('row_partition must be a RowPartition.')
        if self.shape.rank == 0:
            raise ValueError('Shape %s must have rank at least 1' % self.shape)
        return _partition_outer_dimension(self, row_partition)

    def merge_dims(self, outer_axis, inner_axis):
        """Merges outer_axis...inner_axis into a single dimension.

    Returns a copy of this RaggedTensor with the specified range of dimensions
    flattened into a single dimension, with elements in row-major order.

    >>> st = tf.experimental.StructuredTensor.from_pyval(
    ...     [[{'foo': 12}, {'foo': 33}], [], [{'foo': 99}]])
    >>> st.merge_dims(0, 1)
    <StructuredTensor(
      fields={
        "foo": tf.Tensor([12 33 99], shape=(3,), dtype=int32)},
      shape=(3,))>

    Args:
      outer_axis: `int`: The first dimension in the range of dimensions to
        merge. May be negative (to index from the last dimension).
      inner_axis: `int`: The last dimension in the range of dimensions to merge.
        May be negative (to index from the last dimension).

    Returns:
      A copy of this tensor, with the specified dimensions merged into a
      single dimension.  The shape of the returned tensor will be
      `self.shape[:outer_axis] + [N] + self.shape[inner_axis + 1:]`, where `N`
      is the total number of slices in the merged dimensions.
    """
        outer_axis = array_ops.get_positive_axis(outer_axis, self.shape.rank, axis_name='outer_axis', ndims_name='rank(self)')
        inner_axis = array_ops.get_positive_axis(inner_axis, self.shape.rank, axis_name='inner_axis', ndims_name='rank(self)')
        if not outer_axis <= inner_axis:
            raise ValueError('Expected outer_axis (%d) to be less than or equal to inner_axis (%d)' % (outer_axis, inner_axis))
        return _merge_dims(self, outer_axis, inner_axis)

    class Spec:
        """A spec for StructuredTensor."""

        def __validate__(self):
            assert self._ragged_shape is not None

        @classmethod
        def _from_fields_and_rank(cls, fields, rank):
            """Creates a spec of a StructuredTensor with fields and rank."""
            shape = None
            for k, v in fields.items():
                field_shape_untruncated = _dynamic_ragged_shape_spec_from_spec(v)
                if field_shape_untruncated is None:
                    raise ValueError(f'Cannot convert spec of {k}.')
                untruncated_rank = field_shape_untruncated.rank
                if untruncated_rank is not None and untruncated_rank < rank:
                    raise ValueError(f'Rank of field {k} is {untruncated_rank}, but must be at least {rank}.')
                field_shape = field_shape_untruncated._truncate(rank)
                if shape is None:
                    shape = field_shape
                else:
                    shape = shape._merge_with(field_shape)
            return StructuredTensor.Spec(_ragged_shape=shape, _fields=fields)

        @classmethod
        def _from_shape(cls, shape: dynamic_ragged_shape.DynamicRaggedShape) -> 'StructuredTensor.Spec':
            """Creates the spec of an empty StructuredTensor."""
            return StructuredTensor.Spec(_ragged_shape=shape, _fields={})

        @property
        def _shape(self) -> tensor_shape.TensorShape:
            return self._ragged_shape._to_tensor_shape()

        @property
        def _field_specs(self) -> Dict[str, type_spec.TypeSpec]:
            return self._fields

        @property
        def shape(self) -> tensor_shape.TensorShape:
            return self._shape

        @property
        def rank(self):
            return self._ragged_shape.rank