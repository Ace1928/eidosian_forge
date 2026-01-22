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
def _lookup_dependency(self, name, cached_dependencies=None):
    """From Trackable. Find a weight in the current graph."""
    unconditional = super(LossScale, self)._lookup_dependency(name, cached_dependencies)
    if unconditional is not None:
        return unconditional
    if context.executing_eagerly():
        graph_key = None
    else:
        graph = ops.get_default_graph()
        graph_key = graph._graph_key
    return self._weights.get((name, graph_key), None)