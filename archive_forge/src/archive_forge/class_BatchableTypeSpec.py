import abc
import functools
from typing import Any, List, Optional, Sequence, Type
import warnings
import numpy as np
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal
from tensorflow.python.types import trace
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class BatchableTypeSpec(TypeSpec, metaclass=abc.ABCMeta):
    """TypeSpec with a batchable tensor encoding.

  The batchable tensor encoding is a list of `tf.Tensor`s that supports
  batching and unbatching.  In particular, stacking (or unstacking)
  values with the same `TypeSpec` must be equivalent to stacking (or
  unstacking) each of their tensor lists.  Unlike the component encoding
  (returned by `self._to_components)`, the batchable tensor encoding
  may require using encoding/decoding ops.

  If a subclass's batchable tensor encoding is not simply a flattened version
  of the component encoding, then the subclass must override `_to_tensor_list`,
  `_from_tensor_list`, and _flat_tensor_specs`.
  """
    __slots__ = []
    __batch_encoder__ = LegacyTypeSpecBatchEncoder()

    @abc.abstractmethod
    def _batch(self, batch_size) -> TypeSpec:
        """Returns a TypeSpec representing a batch of objects with this TypeSpec.

    Args:
      batch_size: An `int` representing the number of elements in a batch, or
        `None` if the batch size may vary.

    Returns:
      A `TypeSpec` representing a batch of objects with this TypeSpec.
    """
        raise NotImplementedError(f'{type(self).__name__}._batch')

    @abc.abstractmethod
    def _unbatch(self) -> TypeSpec:
        """Returns a TypeSpec representing a single element this TypeSpec.

    Returns:
      A `TypeSpec` representing a single element of objects with this TypeSpec.
    """
        raise NotImplementedError(f'{type(self).__name__}._unbatch')

    @property
    def _flat_tensor_specs(self) -> List[TypeSpec]:
        """A list of TensorSpecs compatible with self._to_tensor_list(v)."""
        component_flat_tensor_specs = nest.map_structure(functools.partial(get_batchable_flat_tensor_specs, context_spec=self), self._component_specs)
        return nest.flatten(component_flat_tensor_specs)

    def _to_tensor_list(self, value: composite_tensor.CompositeTensor) -> List['core_types.Symbol']:
        """Encodes `value` as a flat list of `core.Symbol`."""
        component_tensor_lists = nest.map_structure(batchable_to_tensor_list, self._component_specs, self._to_components(value))
        return nest.flatten(component_tensor_lists)

    def _to_batched_tensor_list(self, value: composite_tensor.CompositeTensor) -> List['core_types.Symbol']:
        """Encodes `value` as a flat list of `core.Symbol` each with rank>0."""
        get_spec_tensor_list = lambda spec, v: batchable_to_tensor_list(spec, v, minimum_rank=1) if isinstance(spec, BatchableTypeSpec) else spec._to_tensor_list(v)
        component_batched_tensor_lists = nest.map_structure(get_spec_tensor_list, self._component_specs, self._to_components(value))
        tensor_list = nest.flatten(component_batched_tensor_lists)
        if any((t.shape.ndims == 0 for t in tensor_list)):
            raise ValueError(f'While converting {value} to a list of tensors for batching, found a scalar item which cannot be batched.')
        return tensor_list

    def _from_compatible_tensor_list(self, tensor_list: List['core_types.Symbol']) -> composite_tensor.CompositeTensor:
        """Reconstructs a value from a compatible flat list of `core.Symbol`."""
        flat_specs = nest.map_structure(functools.partial(get_batchable_flat_tensor_specs, context_spec=self), self._component_specs)
        nested_tensor_list = nest.pack_sequence_as(flat_specs, tensor_list)
        components = nest.map_structure_up_to(self._component_specs, batchable_from_tensor_list, self._component_specs, nested_tensor_list)
        return self._from_components(components)