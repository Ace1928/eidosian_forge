import abc
import functools
import queue
import threading
import warnings
import numpy as np
from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['data.Dataset'])
class DatasetV1(DatasetV2, data_types.DatasetV1):
    """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements and a "logical plan" of transformations that act on
  those elements.
  """

    def __init__(self):
        try:
            variant_tensor = self._as_variant_tensor()
        except AttributeError as e:
            if '_as_variant_tensor' in str(e):
                raise AttributeError('Please use `_variant_tensor` instead of `_as_variant_tensor()` to obtain the variant associated with a dataset.')
            raise AttributeError('{}: A likely cause of this error is that the super call for this dataset is not the last line of the `__init__` method. The base class invokes the `_as_variant_tensor()` method in its constructor and if that method uses attributes defined in the `__init__` method, those attributes need to be defined before the super call.'.format(e))
        super(DatasetV1, self).__init__(variant_tensor)

    @abc.abstractmethod
    def _as_variant_tensor(self):
        """Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.

    Returns:
      A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.
    """
        raise NotImplementedError(f'{type(self)}.as_variant_tensor()')

    @deprecation.deprecated(None, 'This is a deprecated API that should only be used in TF 1 graph mode and legacy TF 2 graph mode available through `tf.compat.v1`. In all other situations -- namely, eager mode and inside `tf.function` -- you can consume dataset elements using `for elem in dataset: ...` or by explicitly creating iterator via `iterator = iter(dataset)` and fetching its elements via `values = next(iterator)`. Furthermore, this API is not available in TF 2. During the transition from TF 1 to TF 2 you can use `tf.compat.v1.data.make_one_shot_iterator(dataset)` to create a TF 1 graph mode style iterator for a dataset created through TF 2 APIs. Note that this should be a transient state of your code base as there are in general no guarantees about the interoperability of TF 1 and TF 2 code.')
    def make_one_shot_iterator(self):
        """Creates an iterator for elements of this dataset.

    Note: The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization. For
    that see `make_initializable_iterator`.

    Example:

    ```python
    # Building graph ...
    dataset = ...
    next_value = dataset.make_one_shot_iterator().get_next()

    # ... from within a session ...
    try:
      while True:
        value = sess.run(next_value)
        ...
    except tf.errors.OutOfRangeError:
        pass
    ```

    Returns:
      An `tf.data.Iterator` for elements of this dataset.
    """
        return self._make_one_shot_iterator()

    def _make_one_shot_iterator(self):
        if context.executing_eagerly():
            with ops.colocate_with(self._variant_tensor):
                return iterator_ops.OwnedIterator(self)
        _ensure_same_dataset_graph(self)
        allowlisted_stateful_ops = traverse.obtain_capture_by_value_ops(self)
        graph_level_seed, op_level_seed = core_random_seed.get_seed(None)

        @function.Defun(capture_by_value=True, allowlisted_stateful_ops=allowlisted_stateful_ops)
        def _make_dataset():
            """Factory function for a dataset."""
            if graph_level_seed is not None:
                assert op_level_seed is not None
                core_random_seed.set_random_seed((graph_level_seed + 89284321 * op_level_seed) % (2 ** 63 - 1))
            dataset = self._apply_debug_options()
            return dataset._variant_tensor
        try:
            _make_dataset.add_to_graph(ops.get_default_graph())
        except ValueError as err:
            if 'Cannot capture a stateful node' in str(err):
                raise ValueError('{}: A likely cause of this error is that the dataset for which you are calling `make_one_shot_iterator()` captures a stateful object, such as a `tf.Variable` or `tf.lookup.StaticHashTable`, which is not supported. Use `make_initializable_iterator()` instead.'.format(err)) from None
            else:
                raise
        with ops.colocate_with(self._variant_tensor):
            return iterator_ops.Iterator(gen_dataset_ops.one_shot_iterator(dataset_factory=_make_dataset, **self._flat_structure), None, get_legacy_output_types(self), get_legacy_output_shapes(self), get_legacy_output_classes(self))

    @deprecation.deprecated(None, 'This is a deprecated API that should only be used in TF 1 graph mode and legacy TF 2 graph mode available through `tf.compat.v1`. In all other situations -- namely, eager mode and inside `tf.function` -- you can consume dataset elements using `for elem in dataset: ...` or by explicitly creating iterator via `iterator = iter(dataset)` and fetching its elements via `values = next(iterator)`. Furthermore, this API is not available in TF 2. During the transition from TF 1 to TF 2 you can use `tf.compat.v1.data.make_initializable_iterator(dataset)` to create a TF 1 graph mode style iterator for a dataset created through TF 2 APIs. Note that this should be a transient state of your code base as there are in general no guarantees about the interoperability of TF 1 and TF 2 code.')
    def make_initializable_iterator(self, shared_name=None):
        """Creates an iterator for elements of this dataset.

    Note: The returned iterator will be in an uninitialized state,
    and you must run the `iterator.initializer` operation before using it:

    ```python
    # Building graph ...
    dataset = ...
    iterator = dataset.make_initializable_iterator()
    next_value = iterator.get_next()  # This is a Tensor.

    # ... from within a session ...
    sess.run(iterator.initializer)
    try:
      while True:
        value = sess.run(next_value)
        ...
    except tf.errors.OutOfRangeError:
        pass
    ```

    Args:
      shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the same
        devices (e.g. when using a remote server).

    Returns:
      A `tf.data.Iterator` for elements of this dataset.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
        return self._make_initializable_iterator(shared_name)

    def _make_initializable_iterator(self, shared_name=None):
        if context.executing_eagerly():
            raise RuntimeError('`make_initializable_iterator()` is not supported in eager mode. Use Python-style iteration instead.')
        _ensure_same_dataset_graph(self)
        dataset = self._apply_debug_options()
        if shared_name is None:
            shared_name = ''
        with ops.colocate_with(self._variant_tensor):
            iterator_resource = gen_dataset_ops.iterator_v2(container='', shared_name=shared_name, **self._flat_structure)
            initializer = gen_dataset_ops.make_iterator(dataset._variant_tensor, iterator_resource)
            return iterator_ops.Iterator(iterator_resource, initializer, get_legacy_output_types(dataset), get_legacy_output_shapes(dataset), get_legacy_output_classes(dataset))

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_classes(dataset)`.')
    def output_classes(self):
        """Returns the class of each component of an element of this dataset.

    Returns:
      A (nested) structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), self.element_spec)

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_shapes(dataset)`.')
    def output_shapes(self):
        """Returns the shape of each component of an element of this dataset.

    Returns:
      A (nested) structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), self.element_spec)

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_types(dataset)`.')
    def output_types(self):
        """Returns the type of each component of an element of this dataset.

    Returns:
      A (nested) structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), self.element_spec)

    @property
    def element_spec(self):
        return structure.convert_legacy_structure(self.output_types, self.output_shapes, self.output_classes)

    @staticmethod
    @functools.wraps(DatasetV2.from_tensors)
    def from_tensors(tensors, name=None):
        return DatasetV1Adapter(DatasetV2.from_tensors(tensors, name=name))

    @staticmethod
    @functools.wraps(DatasetV2.from_tensor_slices)
    def from_tensor_slices(tensors, name=None):
        return DatasetV1Adapter(DatasetV2.from_tensor_slices(tensors, name=name))

    @staticmethod
    @deprecation.deprecated(None, 'Use `tf.data.Dataset.from_tensor_slices()`.')
    def from_sparse_tensor_slices(sparse_tensor):
        """Splits each rank-N `tf.sparse.SparseTensor` in this dataset row-wise.

    Args:
      sparse_tensor: A `tf.sparse.SparseTensor`.

    Returns:
      Dataset: A `Dataset` of rank-(N-1) sparse tensors.
    """
        from tensorflow.python.data.ops import from_sparse_tensor_slices_op
        return from_sparse_tensor_slices_op._from_sparse_tensor_slices(sparse_tensor)

    @staticmethod
    @functools.wraps(DatasetV2.from_generator)
    @deprecation.deprecated_args(None, 'Use output_signature instead', 'output_types', 'output_shapes')
    def from_generator(generator, output_types=None, output_shapes=None, args=None, output_signature=None, name=None):
        with deprecation.silence():
            return DatasetV1Adapter(DatasetV2.from_generator(generator, output_types, output_shapes, args, output_signature, name=name))

    @staticmethod
    @functools.wraps(DatasetV2.range)
    def range(*args, **kwargs):
        return DatasetV1Adapter(DatasetV2.range(*args, **kwargs))

    @staticmethod
    @functools.wraps(DatasetV2.zip)
    def zip(*args, datasets=None, name=None):
        return DatasetV1Adapter(DatasetV2.zip(*args, datasets=datasets, name=name))

    @functools.wraps(DatasetV2.concatenate)
    def concatenate(self, dataset, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).concatenate(dataset, name=name))

    @functools.wraps(DatasetV2.prefetch)
    def prefetch(self, buffer_size, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).prefetch(buffer_size, name=name))

    @staticmethod
    @functools.wraps(DatasetV2.list_files)
    def list_files(file_pattern, shuffle=None, seed=None, name=None):
        return DatasetV1Adapter(DatasetV2.list_files(file_pattern, shuffle, seed, name=name))

    @functools.wraps(DatasetV2.repeat)
    def repeat(self, count=None, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).repeat(count, name=name))

    @functools.wraps(DatasetV2.shuffle)
    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).shuffle(buffer_size, seed, reshuffle_each_iteration, name=name))

    @functools.wraps(DatasetV2.cache)
    def cache(self, filename='', name=None):
        return DatasetV1Adapter(super(DatasetV1, self).cache(filename, name=name))

    @functools.wraps(DatasetV2.take)
    def take(self, count, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).take(count, name=name))

    @functools.wraps(DatasetV2.skip)
    def skip(self, count, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).skip(count, name=name))

    @functools.wraps(DatasetV2.shard)
    def shard(self, num_shards, index, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).shard(num_shards, index, name=name))

    @functools.wraps(DatasetV2.batch)
    def batch(self, batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).batch(batch_size, drop_remainder, num_parallel_calls, deterministic, name=name))

    @functools.wraps(DatasetV2.padded_batch)
    def padded_batch(self, batch_size, padded_shapes=None, padding_values=None, drop_remainder=False, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).padded_batch(batch_size, padded_shapes, padding_values, drop_remainder, name=name))

    @functools.wraps(DatasetV2.map)
    def map(self, map_func, num_parallel_calls=None, deterministic=None, name=None):
        from tensorflow.python.data.ops import map_op
        return map_op._map_v1(self, map_func, num_parallel_calls=num_parallel_calls, deterministic=deterministic)

    @deprecation.deprecated(None, 'Use `tf.data.Dataset.map()')
    def map_with_legacy_function(self, map_func, num_parallel_calls=None, deterministic=None):
        """Maps `map_func` across the elements of this dataset.

    Note: This is an escape hatch for existing uses of `map` that do not work
    with V2 functions. New uses are strongly discouraged and existing uses
    should migrate to `map` as this method will be removed in V2.

    Args:
      map_func: A function mapping a (nested) structure of tensors (having
        shapes and types defined by `self.output_shapes` and
        `self.output_types`) to another (nested) structure of tensors.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process asynchronously in parallel.
        If not specified, elements will be processed sequentially. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel calls is set
        dynamically based on available CPU.
      deterministic: (Optional.) When `num_parallel_calls` is specified, this
        boolean controls the order in which the transformation produces
        elements. If set to `False`, the transformation is allowed to yield
        elements out of order to trade determinism for performance. If not
        specified, the `tf.data.Options.deterministic` option (`True` by
        default) controls the behavior.

    Returns:
      Dataset: A `Dataset`.
    """
        from tensorflow.python.data.ops import map_op
        return map_op._map_v1_with_legacy_function(self, map_func, num_parallel_calls=num_parallel_calls, deterministic=deterministic)

    @functools.wraps(DatasetV2.flat_map)
    def flat_map(self, map_func, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).flat_map(map_func, name=name))

    @functools.wraps(DatasetV2.interleave)
    def interleave(self, map_func, cycle_length=None, block_length=None, num_parallel_calls=None, deterministic=None, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).interleave(map_func, cycle_length, block_length, num_parallel_calls, deterministic, name=name))

    @functools.wraps(DatasetV2.filter)
    def filter(self, predicate, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).filter(predicate, name=name))

    @deprecation.deprecated(None, 'Use `tf.data.Dataset.filter()')
    def filter_with_legacy_function(self, predicate):
        """Filters this dataset according to `predicate`.

    Note: This is an escape hatch for existing uses of `filter` that do not work
    with V2 functions. New uses are strongly discouraged and existing uses
    should migrate to `filter` as this method will be removed in V2.

    Args:
      predicate: A function mapping a (nested) structure of tensors (having
        shapes and types defined by `self.output_shapes` and
        `self.output_types`) to a scalar `tf.bool` tensor.

    Returns:
      Dataset: The `Dataset` containing the elements of this dataset for which
          `predicate` is `True`.
    """
        from tensorflow.python.data.ops import filter_op
        return filter_op._FilterDataset(self, predicate, use_legacy_function=True)

    @functools.wraps(DatasetV2.apply)
    def apply(self, transformation_func):
        return DatasetV1Adapter(super(DatasetV1, self).apply(transformation_func))

    @functools.wraps(DatasetV2.window)
    def window(self, size, shift=None, stride=1, drop_remainder=False, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).window(size, shift, stride, drop_remainder, name=name))

    @functools.wraps(DatasetV2.unbatch)
    def unbatch(self, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).unbatch(name=name))

    @functools.wraps(DatasetV2.with_options)
    def with_options(self, options, name=None):
        return DatasetV1Adapter(super(DatasetV1, self).with_options(options, name=name))