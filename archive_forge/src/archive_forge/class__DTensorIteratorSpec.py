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
class _DTensorIteratorSpec(iterator_ops.IteratorSpec):
    """Type specification for `_DTensorIterator`."""
    __slots__ = ['_global_element_spec', '_layouts_str']

    def __init__(self, global_element_spec: tensor_spec.TensorSpec, layouts_str: Any):
        super().__init__(global_element_spec)
        self._global_element_spec = global_element_spec
        self._layouts_str = layouts_str

    @property
    def value_type(self):
        return _DTensorIterator

    def _serialize(self):
        return (self._global_element_spec, self._layouts_str)

    @property
    def _component_specs(self):
        return (tensor_spec.TensorSpec([], dtypes.resource),)

    def _to_components(self, value):
        return (value._iterator_resource_dtensor,)

    def _from_components(self, components):
        layouts = nest.map_structure(layout_lib.Layout.from_string, self._layouts_str)
        return _DTensorIterator(dtensor_components=components, global_element_spec=self._global_element_spec, layouts=layouts)

    @classmethod
    def from_value(cls, value):
        return cls(value._global_element_spec, value._layouts_str)