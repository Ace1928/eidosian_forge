import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
class _LiteAggregateOperand(_LiteOperand):
    """An operand for a tflite hint function that is aggregated from many.

  For example, an LSTM is a grid of operators that are all related. Inputs
  going into them may need to be fused, so they should all be tracked as
  related arguments.
  """

    def __init__(self, aggregation):
        _LiteOperand.__init__(self)
        self.aggregation = aggregation
        self.names = {}
        self.nodes = {}
        self.flattened = None

    def add(self, sort, node):
        self.names[sort] = _tensor_name_base(node.name)
        self.nodes[sort] = node

    def flatten_nodes(self):
        """Return a list of all the node protos in aggregation sorted order."""
        if not self.flattened:
            self.flattened = [None] * len(self.nodes)
            for idx, node in self.nodes.items():
                self.flattened[idx] = node
            for n in self.nodes:
                if n is None:
                    raise RuntimeError('Aggregate was missing argument.')
            if self.aggregation == OpHint.AGGREGATE_FIRST:
                self.flattened = self.flattened[:1]
            elif self.aggregation == OpHint.AGGREGATE_LAST:
                self.flattened = self.flattened[-1:]
            elif self.aggregation == OpHint.AGGREGATE_STACK:
                pass
            else:
                raise ValueError('Invalid aggregation type %r specified' % self.aggregation)
        return self.flattened

    def flatten(self):
        """Return a list of all node names in aggregation sorted sorter."""
        return [_tensor_name_base(x.name) for x in self.flatten_nodes()]

    def aggregate_and_return_name_for_input(self, out_graphdef):
        """This adds the nodes to out_graphdef and returns an aggregated output.

    In particular, if you have 4 inputs to a hint stub, this will be the
    node that you can use as an output. I.e. you have 4 timesteps from a
    static rnn, then a fused UnidirectionalLSTM will expect 1 input with
    all 4 time steps. So here we make a pack and return the output name of
    that pack.

    Args:
      out_graphdef: A graphdef that is ready to have this input added.

    Returns:
      The name of a pack that aggregates this node.
    """
        flattened = self.flatten_nodes()
        if self.aggregation == OpHint.AGGREGATE_FIRST or self.aggregation == OpHint.AGGREGATE_LAST:
            assert len(flattened) == 1
        if len(flattened) == 1 and self.aggregation != OpHint.AGGREGATE_STACK:
            return _tensor_name_base(flattened[0].name)
        else:
            new_node = _node_def_pb2.NodeDef()
            new_node.op = 'Pack'
            new_node.name = 'OpHintStack-%s' % flattened[0].name
            new_node.attr['N'].i = len(flattened)
            new_node.attr['T'].type = flattened[0].attr['T'].type
            for discrete in flattened:
                new_node.input.append(_tensor_name_base(discrete.name))
            out_graphdef.node.extend([new_node])
            return new_node.name

    def aggregate_and_return_name_for_output(self, fused_op_name, output_index, out_graphdef):
        """This adds to `out_graphdef` all the unaggregated outputs.

    I.e. we are outputting from a fused stub, but we need to make it compatible
    with the unfused original graph so we insert an unpack. Ideally in a later
    stage the unpack -> pack sequences will be removed.

    Args:
      fused_op_name: The name of the stub we are in the process of fusing.
      output_index: The output output_index this object represents.
      out_graphdef: The graphdef we are in the process of buildings

    Returns:
      The type of the aggregated output (so we can finish building the stub
      op).
    """
        flattened = self.flatten_nodes()
        if self.aggregation == OpHint.AGGREGATE_FIRST or self.aggregation == OpHint.AGGREGATE_LAST:
            assert len(flattened) == 1
        if len(flattened) == 1 and self.aggregation != OpHint.AGGREGATE_STACK:
            temp_op = _LiteSingleOperand(flattened[0])
            return temp_op.aggregate_and_return_name_for_output(fused_op_name, output_index, out_graphdef)
        else:
            stack_node = _node_def_pb2.NodeDef()
            stack_node.op = 'Unpack'
            stack_node.name = 'OpHintUnstack-%s' % flattened[0].name
            stack_node.attr['num'].i = len(flattened)
            output_type = flattened[0].attr['T'].type
            stack_node.attr['T'].type = output_type
            stack_node.input.append(_tensorflow_output_name(fused_op_name, output_index))
            out_graphdef.node.extend([stack_node])
            for idx, discrete in enumerate(flattened):
                output_node = _copy.deepcopy(discrete)
                del output_node.input[:]
                output_node.input.append(_tensorflow_output_name(stack_node.name, idx))
                out_graphdef.node.extend([output_node])
            return output_type

    def __str__(self):
        s = '\t\t\tAGGREGATE %s\n' % self.aggregation
        for sort, val in self.names.iteritems():
            s += '\t\t\t%d: %s\n' % (sort, val)
        return s