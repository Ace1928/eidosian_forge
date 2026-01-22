import abc
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _add_weight(self, name, initial_value, dtype=None):
    """Adds a weight to this loss scale.

    Args:
      name: Variable name.
      initial_value: The variable's initial value.
      dtype: The type of the variable.

    Returns:
      A variable.

    Raises:
      RuntimeError: If a weight with `name` has already been added.
    """
    variable = variable_v1.VariableV1(initial_value=initial_value, name=name, dtype=dtype, trainable=False, use_resource=True, synchronization=variables.VariableSynchronization.AUTO, aggregation=variables.VariableAggregation.NONE)
    if context.executing_eagerly():
        graph_key = None
    else:
        graph = ops.get_default_graph()
        graph_key = graph._graph_key
    key = (name, graph_key)
    if self._weights.get(key, None) is not None:
        raise RuntimeError('Duplicate variables detected. {}'.format(key))
    self._weights[key] = variable
    self._handle_deferred_dependencies(name=name, trackable=variable)
    return variable