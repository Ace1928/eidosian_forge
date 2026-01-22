from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def _prune_non_control_edges_of_debug_ops(self):
    """Prune (non-control) edges related to debug ops.

    Prune the Copy ops and associated _Send ops inserted by the debugger out
    from the non-control inputs and output recipients map. Replace the inputs
    and recipients with original ones.
    """
    for node in self._node_inputs:
        inputs = self._node_inputs[node]
        for i, inp in enumerate(inputs):
            if is_copy_node(inp):
                orig_inp = self._node_inputs[inp][0]
                inputs[i] = orig_inp