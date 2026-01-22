import threading
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec as type_spec_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class PerWorkerValuesTypeSpec(type_spec_lib.TypeSpec):
    """TypeSpec for PerWorkerValues.

  It only support tracing a function using a PerWorkerValues.
  """

    def __init__(self, value_spec, descendant_type):
        assert value_spec
        self._value_spec = value_spec
        self._descendant_type = descendant_type

    def _serialize(self):
        return (self._value_spec,)

    @property
    def value_type(self):
        return self._descendant_type

    def most_specific_common_supertype(self, others):
        raise NotImplementedError('most_specific_common_supertype is not implemented')

    @property
    def _component_specs(self):
        return self._value_spec

    def _to_components(self, value):
        return self._value_spec

    def _from_components(self, value):
        return value