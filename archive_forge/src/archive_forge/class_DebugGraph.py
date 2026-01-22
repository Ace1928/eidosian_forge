from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
class DebugGraph(object):
    """Represents a debugger-decorated graph."""

    def __init__(self, debug_graph_def, device_name=None):
        self._debug_graph_def = debug_graph_def
        self._non_debug_graph_def = None
        self._node_attributes = {}
        self._node_inputs = {}
        self._node_reversed_ref_inputs = {}
        self._node_ctrl_inputs = {}
        self._node_recipients = {}
        self._node_ctrl_recipients = {}
        self._node_devices = {}
        self._node_op_types = {}
        self._copy_send_nodes = []
        self._ref_args = {}
        self._device_name = device_name
        if not self._device_name:
            self._device_name = _infer_device_name(debug_graph_def)
        for node in debug_graph_def.node:
            self._process_debug_graph_node(node)
        self._prune_non_control_edges_of_debug_ops()
        self._prune_control_edges_of_debug_ops()
        self._prune_nodes_from_input_and_recipient_maps(self._get_copy_nodes())
        self._populate_recipient_maps()

    def _process_debug_graph_node(self, node):
        """Process a node from the debug GraphDef.

    Args:
      node: (NodeDef) A partition-graph node to be processed.

    Raises:
      ValueError: If duplicate node names are encountered.
    """
        if is_debug_node(node.name):
            return
        if node.name in self._node_inputs:
            raise ValueError("Duplicate node name on device %s: '%s'" % (self._device_name, node.name))
        self._node_attributes[node.name] = node.attr
        self._node_inputs[node.name] = []
        self._node_ctrl_inputs[node.name] = []
        self._node_recipients[node.name] = []
        self._node_ctrl_recipients[node.name] = []
        if node.name not in self._node_devices:
            self._node_devices[node.name] = set()
        self._node_devices[node.name].add(node.device if node.device else self._device_name)
        self._node_op_types[node.name] = node.op
        self._ref_args[node.name] = self._get_ref_args(node)
        for inp in node.input:
            if is_copy_node(inp) and (node.op == '_Send' or node.op == '_Retval'):
                self._copy_send_nodes.append(node.name)
            if inp.startswith('^'):
                cinp = inp[1:]
                self._node_ctrl_inputs[node.name].append(cinp)
            else:
                self._node_inputs[node.name].append(inp)

    def _get_ref_args(self, node):
        """Determine whether an input of an op is ref-type.

    Args:
      node: A `NodeDef`.

    Returns:
      A list of the arg names (as strs) that are ref-type.
    """
        op_def = op_def_registry.get(node.op)
        if op_def is None:
            return []
        ref_args = []
        for i, output_arg in enumerate(op_def.output_arg):
            if output_arg.is_ref:
                arg_name = node.name if i == 0 else '%s:%d' % (node.name, i)
                ref_args.append(arg_name)
        return ref_args

    def _get_copy_nodes(self):
        """Find all Copy nodes in the loaded graph."""
        copy_nodes = []
        for node in self._node_inputs:
            if is_copy_node(node):
                copy_nodes.append(node)
        return copy_nodes

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

    def _prune_control_edges_of_debug_ops(self):
        """Prune control edges related to the debug ops."""
        for node in self._node_ctrl_inputs:
            ctrl_inputs = self._node_ctrl_inputs[node]
            debug_op_inputs = []
            for ctrl_inp in ctrl_inputs:
                if is_debug_node(ctrl_inp):
                    debug_op_inputs.append(ctrl_inp)
            for debug_op_inp in debug_op_inputs:
                ctrl_inputs.remove(debug_op_inp)

    def _populate_recipient_maps(self):
        """Populate the map from node name to recipient(s) of its output(s).

    This method also populates the input map based on reversed ref edges.
    """
        for node in self._node_inputs:
            inputs = self._node_inputs[node]
            for inp in inputs:
                inp = get_node_name(inp)
                if inp not in self._node_recipients:
                    self._node_recipients[inp] = []
                self._node_recipients[inp].append(node)
                if inp in self._ref_args:
                    if inp not in self._node_reversed_ref_inputs:
                        self._node_reversed_ref_inputs[inp] = []
                    self._node_reversed_ref_inputs[inp].append(node)
        for node in self._node_ctrl_inputs:
            ctrl_inputs = self._node_ctrl_inputs[node]
            for ctrl_inp in ctrl_inputs:
                if ctrl_inp in self._copy_send_nodes:
                    continue
                if ctrl_inp not in self._node_ctrl_recipients:
                    self._node_ctrl_recipients[ctrl_inp] = []
                self._node_ctrl_recipients[ctrl_inp].append(node)

    def _prune_nodes_from_input_and_recipient_maps(self, nodes_to_prune):
        """Prune nodes out of input and recipient maps.

    Args:
      nodes_to_prune: (`list` of `str`) Names of the nodes to be pruned.
    """
        for node in nodes_to_prune:
            del self._node_inputs[node]
            del self._node_ctrl_inputs[node]
            del self._node_recipients[node]
            del self._node_ctrl_recipients[node]

    def _reconstruct_non_debug_graph_def(self):
        """Reconstruct non-debug GraphDef.

    Non-debug GraphDef means the original GraphDef without the Copy* and Debug
    nodes inserted by the debugger.
    """
        if self._non_debug_graph_def:
            return
        self._non_debug_graph_def = graph_pb2.GraphDef()
        for node in self._debug_graph_def.node:
            if is_copy_node(node.name) or is_debug_node(node.name):
                continue
            new_node = self._non_debug_graph_def.node.add()
            new_node.CopyFrom(node)
            del new_node.input[:]
            for inp in self._node_inputs[node.name]:
                new_node.input.append(inp)
            for ctrl_inp in self._node_ctrl_inputs[node.name]:
                new_node.input.append('^' + ctrl_inp)

    @property
    def device_name(self):
        return self._device_name

    @property
    def debug_graph_def(self):
        """The debugger-decorated GraphDef."""
        return self._debug_graph_def

    @property
    def non_debug_graph_def(self):
        """The GraphDef without the Copy* and Debug* nodes added by the debugger."""
        self._reconstruct_non_debug_graph_def()
        return self._non_debug_graph_def

    @property
    def node_devices(self):
        return self._node_devices

    @property
    def node_op_types(self):
        return self._node_op_types

    @property
    def node_attributes(self):
        return self._node_attributes

    @property
    def node_inputs(self):
        return self._node_inputs

    @property
    def node_ctrl_inputs(self):
        return self._node_ctrl_inputs

    @property
    def node_reversed_ref_inputs(self):
        return self._node_reversed_ref_inputs

    @property
    def node_recipients(self):
        return self._node_recipients

    @property
    def node_ctrl_recipients(self):
        return self._node_ctrl_recipients