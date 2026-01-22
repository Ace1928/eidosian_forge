import dataclasses
import operator
from typing import Any, List, Optional, Sequence, Tuple
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class _DTensorIterator(iterator_ops.OwnedIterator):
    """An iterator for a tf.data.Dataset distributed using DTensor.

  DTensorIterator encapsulates multiple underlying dataset iterators. It handles
  retrieving the tensors to be placed on each underlying device and then uses
  the 'pack' operation to create and return a DTensor. Thus users need only
  interact with a single DTensorIterator to automatically distribute dataset
  tensors onto devices.
  """

    def __init__(self, dtensor_components: Tuple[tensor.Tensor], global_element_spec: tensor_spec.TensorSpec, layouts: Any):
        """Initializes a distributed iterator for DTensor datasets.

    This iterator encapsulates tf.data iterators for the underlying devices, and
    treats it as a packed DTensor of iterator resource tensors.

    Args:
      dtensor_components: a tuple containing the underlying iterator resources
        packed into a DTensor. This is expected to be a tuple with a single
        element.
      global_element_spec: the underlying dataset's element spec from a global
        view.
      layouts: a structure of DTensor layouts to be applied to the elements
        returned by the underlying iterators. This can be a single layout or
        (possibly nested) tuples or dictionaries of layouts, and the structure
        must match the structure of the iterator elements.
    """
        [self._iterator_resource_dtensor] = dtensor_components
        self._global_element_spec = global_element_spec
        self._layouts = layouts
        self._layouts_str = nest.map_structure(lambda layout: layout.to_string(), layouts)
        super().__init__(components=dtensor_components, element_spec=global_element_spec)

    def __next__(self):
        try:
            host_elem = self._next_internal()
            context.async_wait()
            device_elem = nest.map_structure(api.copy_to_mesh, host_elem, self._layouts)
            context.async_wait()
            return device_elem
        except errors.OutOfRangeError as e:
            if context.executing_eagerly():
                raise StopIteration from e
            else:
                raise e

    @property
    def _type_spec(self):
        return _DTensorIteratorSpec(self._global_element_spec, self._layouts_str)