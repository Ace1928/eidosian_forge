from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _get_global_step_read(graph=None):
    """Gets global step read tensor in graph.

  Args:
    graph: The graph in which to create the global step read tensor. If missing,
      use default graph.

  Returns:
    Global step read tensor.

  Raises:
    RuntimeError: if multiple items found in collection GLOBAL_STEP_READ_KEY.
  """
    graph = graph or ops.get_default_graph()
    global_step_read_tensors = graph.get_collection(GLOBAL_STEP_READ_KEY)
    if len(global_step_read_tensors) > 1:
        raise RuntimeError('There are multiple items in collection {}. There should be only one.'.format(GLOBAL_STEP_READ_KEY))
    if len(global_step_read_tensors) == 1:
        return global_step_read_tensors[0]
    return None