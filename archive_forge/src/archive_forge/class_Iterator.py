import abc
import threading
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.data.ops import iterator_autograph
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@saveable_compat.legacy_saveable_name('ITERATOR')
@tf_export(v1=['data.Iterator'])
class Iterator(trackable.Trackable):
    """Represents the state of iterating through a `Dataset`."""

    def __init__(self, iterator_resource, initializer, output_types, output_shapes, output_classes):
        """Creates a new iterator from the given iterator resource.

    Note: Most users will not call this initializer directly, and will
    instead use `Dataset.make_initializable_iterator()` or
    `Dataset.make_one_shot_iterator()`.

    Args:
      iterator_resource: A `tf.resource` scalar `tf.Tensor` representing the
        iterator.
      initializer: A `tf.Operation` that should be run to initialize this
        iterator.
      output_types: A (nested) structure of `tf.DType` objects corresponding to
        each component of an element of this iterator.
      output_shapes: A (nested) structure of `tf.TensorShape` objects
        corresponding to each component of an element of this iterator.
      output_classes: A (nested) structure of Python `type` objects
        corresponding to each component of an element of this iterator.

    Raises:
      TypeError: If `output_types`, `output_shapes`, or `output_classes` is not
        specified.
    """
        self._iterator_resource = iterator_resource
        self._initializer = initializer
        if output_types is None or output_shapes is None or output_classes is None:
            raise ValueError(f'All of `output_types`, `output_shapes`, and `output_classes` must be specified to create an iterator. Got `output_types` = {output_types!r}, `output_shapes` = {output_shapes!r}, `output_classes` = {output_classes!r}.')
        self._element_spec = structure.convert_legacy_structure(output_types, output_shapes, output_classes)
        self._flat_tensor_shapes = structure.get_flat_tensor_shapes(self._element_spec)
        self._flat_tensor_types = structure.get_flat_tensor_types(self._element_spec)
        self._string_handle = gen_dataset_ops.iterator_to_string_handle(self._iterator_resource)
        self._get_next_call_count = 0
        ops.add_to_collection(GLOBAL_ITERATORS, self._iterator_resource)

    @staticmethod
    def from_structure(output_types, output_shapes=None, shared_name=None, output_classes=None):
        """Creates a new, uninitialized `Iterator` with the given structure.

    This iterator-constructing method can be used to create an iterator that
    is reusable with many different datasets.

    The returned iterator is not bound to a particular dataset, and it has
    no `initializer`. To initialize the iterator, run the operation returned by
    `Iterator.make_initializer(dataset)`.

    The following is an example

    ```python
    iterator = Iterator.from_structure(tf.int64, tf.TensorShape([]))

    dataset_range = Dataset.range(10)
    range_initializer = iterator.make_initializer(dataset_range)

    dataset_evens = dataset_range.filter(lambda x: x % 2 == 0)
    evens_initializer = iterator.make_initializer(dataset_evens)

    # Define a model based on the iterator; in this example, the model_fn
    # is expected to take scalar tf.int64 Tensors as input (see
    # the definition of 'iterator' above).
    prediction, loss = model_fn(iterator.get_next())

    # Train for `num_epochs`, where for each epoch, we first iterate over
    # dataset_range, and then iterate over dataset_evens.
    for _ in range(num_epochs):
      # Initialize the iterator to `dataset_range`
      sess.run(range_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break

      # Initialize the iterator to `dataset_evens`
      sess.run(evens_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break
    ```

    Args:
      output_types: A (nested) structure of `tf.DType` objects corresponding to
        each component of an element of this dataset.
      output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`
        objects corresponding to each component of an element of this dataset.
        If omitted, each component will have an unconstrainted shape.
      shared_name: (Optional.) If non-empty, this iterator will be shared under
        the given name across multiple sessions that share the same devices
        (e.g. when using a remote server).
      output_classes: (Optional.) A (nested) structure of Python `type` objects
        corresponding to each component of an element of this iterator. If
        omitted, each component is assumed to be of type `tf.Tensor`.

    Returns:
      An `Iterator`.

    Raises:
      TypeError: If the structures of `output_shapes` and `output_types` are
        not the same.
    """
        output_types = nest.map_structure(dtypes.as_dtype, output_types)
        if output_shapes is None:
            output_shapes = nest.map_structure(lambda _: tensor_shape.TensorShape(None), output_types)
        else:
            output_shapes = nest.map_structure_up_to(output_types, tensor_shape.as_shape, output_shapes)
        if output_classes is None:
            output_classes = nest.map_structure(lambda _: tensor.Tensor, output_types)
        nest.assert_same_structure(output_types, output_shapes)
        output_structure = structure.convert_legacy_structure(output_types, output_shapes, output_classes)
        if shared_name is None:
            shared_name = ''
        iterator_resource = gen_dataset_ops.iterator_v2(container='', shared_name=shared_name, output_types=structure.get_flat_tensor_types(output_structure), output_shapes=structure.get_flat_tensor_shapes(output_structure))
        return Iterator(iterator_resource, None, output_types, output_shapes, output_classes)

    @staticmethod
    def from_string_handle(string_handle, output_types, output_shapes=None, output_classes=None):
        """Creates a new, uninitialized `Iterator` based on the given handle.

    This method allows you to define a "feedable" iterator where you can choose
    between concrete iterators by feeding a value in a `tf.Session.run` call.
    In that case, `string_handle` would be a `tf.compat.v1.placeholder`, and you
    would
    feed it with the value of `tf.data.Iterator.string_handle` in each step.

    For example, if you had two iterators that marked the current position in
    a training dataset and a test dataset, you could choose which to use in
    each step as follows:

    ```python
    train_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    train_iterator_handle = sess.run(train_iterator.string_handle())

    test_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    test_iterator_handle = sess.run(test_iterator.string_handle())

    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_iterator.output_types)

    next_element = iterator.get_next()
    loss = f(next_element)

    train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
    test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})
    ```

    Args:
      string_handle: A scalar `tf.Tensor` of type `tf.string` that evaluates to
        a handle produced by the `Iterator.string_handle()` method.
      output_types: A (nested) structure of `tf.DType` objects corresponding to
        each component of an element of this dataset.
      output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`
        objects corresponding to each component of an element of this dataset.
        If omitted, each component will have an unconstrainted shape.
      output_classes: (Optional.) A (nested) structure of Python `type` objects
        corresponding to each component of an element of this iterator. If
        omitted, each component is assumed to be of type `tf.Tensor`.

    Returns:
      An `Iterator`.
    """
        output_types = nest.map_structure(dtypes.as_dtype, output_types)
        if output_shapes is None:
            output_shapes = nest.map_structure(lambda _: tensor_shape.TensorShape(None), output_types)
        else:
            output_shapes = nest.map_structure_up_to(output_types, tensor_shape.as_shape, output_shapes)
        if output_classes is None:
            output_classes = nest.map_structure(lambda _: tensor.Tensor, output_types)
        nest.assert_same_structure(output_types, output_shapes)
        output_structure = structure.convert_legacy_structure(output_types, output_shapes, output_classes)
        string_handle = ops.convert_to_tensor(string_handle, dtype=dtypes.string)
        iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(string_handle, output_types=structure.get_flat_tensor_types(output_structure), output_shapes=structure.get_flat_tensor_shapes(output_structure))
        return Iterator(iterator_resource, None, output_types, output_shapes, output_classes)

    @property
    def initializer(self):
        """A `tf.Operation` that should be run to initialize this iterator.

    Returns:
      A `tf.Operation` that should be run to initialize this iterator

    Raises:
      ValueError: If this iterator initializes itself automatically.
    """
        if self._initializer is not None:
            return self._initializer
        else:
            raise ValueError('The iterator does not have an initializer. This means it was likely created using `tf.data.Dataset.make_one_shot_iterator()`. For an initializable iterator, use `tf.data.Dataset.make_initializable_iterator()` instead.')

    def make_initializer(self, dataset, name=None):
        """Returns a `tf.Operation` that initializes this iterator on `dataset`.

    Args:
      dataset: A `Dataset` whose `element_spec` if compatible with this
        iterator.
      name: (Optional.) A name for the created operation.

    Returns:
      A `tf.Operation` that can be run to initialize this iterator on the given
      `dataset`.

    Raises:
      TypeError: If `dataset` and this iterator do not have a compatible
        `element_spec`.
    """
        with ops.name_scope(name, 'make_initializer') as name:
            dataset_output_types = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), dataset.element_spec)
            dataset_output_shapes = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), dataset.element_spec)
            dataset_output_classes = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), dataset.element_spec)
            nest.assert_same_structure(self.output_types, dataset_output_types)
            nest.assert_same_structure(self.output_shapes, dataset_output_shapes)
            for iterator_class, dataset_class in zip(nest.flatten(self.output_classes), nest.flatten(dataset_output_classes)):
                if iterator_class is not dataset_class:
                    raise TypeError(f'Expected output classes {self.output_classes!r} but got dataset with output classes {dataset_output_classes!r}.')
            for iterator_dtype, dataset_dtype in zip(nest.flatten(self.output_types), nest.flatten(dataset_output_types)):
                if iterator_dtype != dataset_dtype:
                    raise TypeError(f'Expected output types {self.output_types!r} but got dataset with output types {dataset_output_types!r}.')
            for iterator_shape, dataset_shape in zip(nest.flatten(self.output_shapes), nest.flatten(dataset_output_shapes)):
                if not iterator_shape.is_compatible_with(dataset_shape):
                    raise TypeError(f'Expected output shapes compatible with {self.output_shapes!r} but got dataset with output shapes {dataset_output_shapes!r}.')
        with ops.colocate_with(self._iterator_resource):
            return gen_dataset_ops.make_iterator(dataset._variant_tensor, self._iterator_resource, name=name)

    def get_next(self, name=None):
        """Returns the next element.

    In graph mode, you should typically call this method *once* and use its
    result as the input to another computation. A typical loop will then call
    `tf.Session.run` on the result of that computation. The loop will terminate
    when the `Iterator.get_next()` operation raises
    `tf.errors.OutOfRangeError`. The following skeleton shows how to use
    this method when building a training loop:

    ```python
    dataset = ...  # A `tf.data.Dataset` object.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Build a TensorFlow graph that does something with each element.
    loss = model_function(next_element)
    optimizer = ...  # A `tf.compat.v1.train.Optimizer` object.
    train_op = optimizer.minimize(loss)

    with tf.compat.v1.Session() as sess:
      try:
        while True:
          sess.run(train_op)
      except tf.errors.OutOfRangeError:
        pass
    ```

    NOTE: It is legitimate to call `Iterator.get_next()` multiple times, e.g.
    when you are distributing different elements to multiple devices in a single
    step. However, a common pitfall arises when users call `Iterator.get_next()`
    in each iteration of their training loop. `Iterator.get_next()` adds ops to
    the graph, and executing each op allocates resources (including threads); as
    a consequence, invoking it in every iteration of a training loop causes
    slowdown and eventual resource exhaustion. To guard against this outcome, we
    log a warning when the number of uses crosses a fixed threshold of
    suspiciousness.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A (nested) structure of values matching `tf.data.Iterator.element_spec`.
    """
        self._get_next_call_count += 1
        if self._get_next_call_count > GET_NEXT_CALL_WARNING_THRESHOLD:
            warnings.warn(GET_NEXT_CALL_WARNING_MESSAGE)
        with ops.colocate_with(self._iterator_resource):
            flat_ret = gen_dataset_ops.iterator_get_next(self._iterator_resource, output_types=self._flat_tensor_types, output_shapes=self._flat_tensor_shapes, name=name)
            return structure.from_tensor_list(self._element_spec, flat_ret)

    def get_next_as_optional(self):
        with ops.colocate_with(self._iterator_resource):
            return optional_ops._OptionalImpl(gen_dataset_ops.iterator_get_next_as_optional(self._iterator_resource, output_types=structure.get_flat_tensor_types(self.element_spec), output_shapes=structure.get_flat_tensor_shapes(self.element_spec)), self.element_spec)

    def string_handle(self, name=None):
        """Returns a string-valued `tf.Tensor` that represents this iterator.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A scalar `tf.Tensor` of type `tf.string`.
    """
        if name is None:
            return self._string_handle
        else:
            return gen_dataset_ops.iterator_to_string_handle(self._iterator_resource, name=name)

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_classes(iterator)`.')
    def output_classes(self):
        """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.sparse.SparseTensor`.

    Returns:
      A (nested) structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), self._element_spec)

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_shapes(iterator)`.')
    def output_shapes(self):
        """Returns the shape of each component of an element of this iterator.

    Returns:
      A (nested) structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), self._element_spec)

    @property
    @deprecation.deprecated(None, 'Use `tf.compat.v1.data.get_output_types(iterator)`.')
    def output_types(self):
        """Returns the type of each component of an element of this iterator.

    Returns:
      A (nested) structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), self._element_spec)

    @property
    def element_spec(self):
        """The type specification of an element of this iterator.

    For more information,
    read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).

    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this iterator and specifying the type of individual components.
    """
        return self._element_spec

    def _serialize_to_tensors(self):
        serialized_iterator = gen_dataset_ops.serialize_iterator(self._iterator_resource, options_lib.ExternalStatePolicy.FAIL.value)
        return {'_STATE': serialized_iterator}

    def _restore_from_tensors(self, restored_tensors):
        with ops.colocate_with(self._iterator_resource):
            return [gen_dataset_ops.deserialize_iterator(self._iterator_resource, restored_tensors['_STATE'])]