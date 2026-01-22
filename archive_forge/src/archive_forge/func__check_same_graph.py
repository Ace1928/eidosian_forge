import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
def _check_same_graph(self, tensor):
    if self._graph is None:
        self._graph = tensor.graph
    elif self._graph != tensor.graph:
        raise ValueError('The graph of the value (%s) is not the same as the graph %s' % (tensor.graph, self._graph))