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
class OwnedIterator(IteratorBase):
    """An iterator producing tf.Tensor objects from a tf.data.Dataset.

  The iterator resource  created through `OwnedIterator` is owned by the Python
  object and the life time of the underlying resource is tied to the life time
  of the `OwnedIterator` object. This makes `OwnedIterator` appropriate for use
  in eager mode and inside of tf.functions.
  """

    def __init__(self, dataset=None, components=None, element_spec=None):
        """Creates a new iterator from the given dataset.

    If `dataset` is not specified, the iterator will be created from the given
    tensor components and element structure. In particular, the alternative for
    constructing the iterator is used when the iterator is reconstructed from
    it `CompositeTensor` representation.

    Args:
      dataset: A `tf.data.Dataset` object.
      components: Tensor components to construct the iterator from.
      element_spec: A (nested) structure of `TypeSpec` objects that
        represents the type specification of elements of the iterator.

    Raises:
      ValueError: If `dataset` is not provided and either `components` or
        `element_spec` is not provided. Or `dataset` is provided and either
        `components` and `element_spec` is provided.
    """
        super(OwnedIterator, self).__init__()
        if dataset is None:
            if components is None or element_spec is None:
                raise ValueError('When `dataset` is not provided, both `components` and `element_spec` must be specified.')
            self._element_spec = element_spec
            self._flat_output_types = structure.get_flat_tensor_types(self._element_spec)
            self._flat_output_shapes = structure.get_flat_tensor_shapes(self._element_spec)
            self._iterator_resource, = components
        else:
            if components is not None or element_spec is not None:
                raise ValueError('When `dataset` is provided, `element_spec` and `components` must not be specified.')
            self._create_iterator(dataset)
        self._get_next_call_count = 0

    def _create_iterator(self, dataset):
        dataset = dataset._apply_debug_options()
        self._dataset = dataset
        ds_variant = dataset._variant_tensor
        self._element_spec = dataset.element_spec
        self._flat_output_types = structure.get_flat_tensor_types(self._element_spec)
        self._flat_output_shapes = structure.get_flat_tensor_shapes(self._element_spec)
        with ops.colocate_with(ds_variant):
            self._iterator_resource = gen_dataset_ops.anonymous_iterator_v3(output_types=self._flat_output_types, output_shapes=self._flat_output_shapes)
            if not context.executing_eagerly():
                fulltype = type_utils.iterator_full_type_from_spec(self._element_spec)
                assert len(fulltype.args[0].args[0].args) == len(self._flat_output_types)
                self._iterator_resource.op.experimental_set_type(fulltype)
            gen_dataset_ops.make_iterator(ds_variant, self._iterator_resource)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def _next_internal(self):
        autograph_status = autograph_ctx.control_status_ctx().status
        autograph_disabled = autograph_status == autograph_ctx.Status.DISABLED
        if not context.executing_eagerly() and autograph_disabled:
            self._get_next_call_count += 1
            if self._get_next_call_count > GET_NEXT_CALL_ERROR_THRESHOLD:
                raise ValueError(GET_NEXT_CALL_ERROR_MESSAGE)
        if not context.executing_eagerly():
            with ops.colocate_with(self._iterator_resource):
                ret = gen_dataset_ops.iterator_get_next(self._iterator_resource, output_types=self._flat_output_types, output_shapes=self._flat_output_shapes)
            return structure.from_compatible_tensor_list(self._element_spec, ret)
        with context.execution_mode(context.SYNC):
            ret = gen_dataset_ops.iterator_get_next(self._iterator_resource, output_types=self._flat_output_types, output_shapes=self._flat_output_shapes)
            try:
                return self._element_spec._from_compatible_tensor_list(ret)
            except AttributeError:
                return structure.from_compatible_tensor_list(self._element_spec, ret)

    def _save(self):
        external_state_policy = None
        if self._dataset and self._dataset.options().experimental_external_state_policy:
            external_state_policy = self._dataset.options().experimental_external_state_policy.value
        state_variant = gen_dataset_ops.serialize_iterator(self._iterator_resource, external_state_policy)
        return parsing_ops.serialize_tensor(state_variant)

    def _restore(self, state):
        state_variant = parsing_ops.parse_tensor(state, dtypes.variant)
        return gen_dataset_ops.deserialize_iterator(self._iterator_resource, state_variant)

    @property
    def _type_spec(self):
        return IteratorSpec(self.element_spec)

    def __next__(self):
        try:
            return self._next_internal()
        except errors.OutOfRangeError:
            raise StopIteration

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
        return self._element_spec

    def get_next(self):
        return self._next_internal()

    def get_next_as_optional(self):
        with ops.colocate_with(self._iterator_resource):
            return optional_ops._OptionalImpl(gen_dataset_ops.iterator_get_next_as_optional(self._iterator_resource, output_types=structure.get_flat_tensor_types(self.element_spec), output_shapes=structure.get_flat_tensor_shapes(self.element_spec)), self.element_spec)

    def _serialize_to_tensors(self):
        serialized_iterator = None
        if self._dataset and self._dataset.options().experimental_external_state_policy:
            serialized_iterator = gen_dataset_ops.serialize_iterator(self._iterator_resource, self._dataset.options().experimental_external_state_policy.value)
        else:
            serialized_iterator = gen_dataset_ops.serialize_iterator(self._iterator_resource, options_lib.ExternalStatePolicy.FAIL.value)
        return {'_STATE': serialized_iterator}

    def _restore_from_tensors(self, restored_tensors):
        with ops.colocate_with(self._iterator_resource):
            return [gen_dataset_ops.deserialize_iterator(self._iterator_resource, restored_tensors['_STATE'])]

    def __tf_tracing_type__(self, _):
        return self._type_spec