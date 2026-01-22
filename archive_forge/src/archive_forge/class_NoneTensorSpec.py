import collections
import functools
import itertools
import wrapt
from tensorflow.python.data.util import nest
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.nest_util import CustomNestProtocol
from tensorflow.python.util.tf_export import tf_export
@type_spec_registry.register('tf.NoneTensorSpec')
class NoneTensorSpec(type_spec.BatchableTypeSpec):
    """Type specification for `None` value."""

    @property
    def value_type(self):
        return NoneTensor

    def _serialize(self):
        return ()

    @property
    def _component_specs(self):
        return []

    def _to_components(self, value):
        return []

    def _from_components(self, components):
        return

    def _to_tensor_list(self, value):
        return []

    @staticmethod
    def from_value(value):
        return NoneTensorSpec()

    def _batch(self, batch_size):
        return NoneTensorSpec()

    def _unbatch(self):
        return NoneTensorSpec()

    def _to_batched_tensor_list(self, value):
        return []

    def _to_legacy_output_types(self):
        return self

    def _to_legacy_output_shapes(self):
        return self

    def _to_legacy_output_classes(self):
        return self

    def most_specific_compatible_shape(self, other):
        if type(self) is not type(other):
            raise ValueError('No `TypeSpec` is compatible with both {} and {}'.format(self, other))
        return self