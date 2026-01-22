import copy
from typing import Optional
import weakref
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.types import trace
class MirroredVariable(DistributedVariable, Mirrored):
    """Holds a map from replica to variables whose values are kept in sync."""

    def _is_mirrored(self):
        return Mirrored._is_mirrored(self)

    def _update_replica(self, update_fn, value, **kwargs):
        return _on_write_update_replica(self, update_fn, value, **kwargs)

    def scatter_min(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_min(*args, **kwargs)
        if self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and self._aggregation != vs.VariableAggregation.NONE:
            raise NotImplementedError(values_util.scatter_error_msg.format(op_name='scatter_min', aggregation=self._aggregation))
        return super(MirroredVariable, self).scatter_min(*args, **kwargs)

    def scatter_max(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_max(*args, **kwargs)
        if self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and self._aggregation != vs.VariableAggregation.NONE:
            raise NotImplementedError(values_util.scatter_error_msg.format(op_name='scatter_max', aggregation=self._aggregation))
        return super(MirroredVariable, self).scatter_max(*args, **kwargs)

    def scatter_update(self, *args, **kwargs):
        if values_util.is_saving_non_distributed():
            return self._primary.scatter_update(*args, **kwargs)
        if self._aggregation != vs.VariableAggregation.ONLY_FIRST_REPLICA and self._aggregation != vs.VariableAggregation.NONE:
            raise NotImplementedError(values_util.scatter_error_msg.format(op_name='scatter_update', aggregation=self._aggregation))
        return super(MirroredVariable, self).scatter_update(*args, **kwargs)

    def _get_cross_replica(self):
        return array_ops.identity(Mirrored._get_cross_replica(self))

    def _as_graph_element(self):
        return self._get_on_device_or_primary()._as_graph_element()

    def _gather_saveables_for_checkpoint(self):
        """Overrides Trackable method.

    This allows both name-based and object-based save and restore of
    MirroredVariables.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """

        def _saveable_factory(name=self._common_name):
            return _MirroredSaveable(self, self._primary, name)
        return {trackable.VARIABLE_VALUE_KEY: _saveable_factory}

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        """Converts a variable to a tensor."""
        if as_ref:
            raise ValueError('You may be using variable created under distribute strategy in TF 1.x control flows. Try explicitly converting the variable to Tensor using variable.read_value(), or switch to TF 2.x.')
        return ops.convert_to_tensor(self._get(), dtype=dtype, name=name, as_ref=as_ref)