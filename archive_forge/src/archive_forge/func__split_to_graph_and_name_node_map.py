from typing import Dict, List, Tuple
from torch.fx import Graph, GraphModule, Node
from torch.fx._compatibility import compatibility
from .matcher_utils import InternalMatch, SubgraphMatcher
def _split_to_graph_and_name_node_map(gm: GraphModule) -> Tuple[GraphModule, Dict[str, Node]]:
    from torch.fx.graph import _PyTreeInfo
    from torch.utils._pytree import tree_flatten, tree_unflatten
    name_node_map = {}
    for n in gm.graph.nodes:
        if n.op == 'output':
            assert gm._out_spec is not None
            output = tree_unflatten(n.args[0], gm._out_spec)
            assert isinstance(output, tuple), 'Expecting the pattern graph to return a tuple'
            assert len(output) >= 2, 'Expecting the pattern graph to have at least two outputs'
            *out, name_node_map = output
            flattened, out_spec = tree_flatten(out)
            assert isinstance(name_node_map, Dict), 'Expecting the input graph to have a dict output as the last element'
            n.args = (flattened,)
            orig_pytree_info = gm._graph._codegen.pytree_info
            gm._graph._codegen.pytree_info = _PyTreeInfo(orig_pytree_info.orig_args, orig_pytree_info.in_spec, out_spec)
    gm.recompile()
    return (gm, name_node_map)