from ray.dag import DAGNode
import os
import tempfile
from ray.dag.utils import _DAGNodeNameGenerator
from ray.util.annotations import DeveloperAPI
def _dag_to_dot(dag: DAGNode):
    """Create a Dot graph from dag.

    TODO(lchu):
    1. add more Dot configs in kwargs,
    e.g. rankdir, alignment, etc.
    2. add more contents to graph,
    e.g. args, kwargs and options of each node

    """
    _check_pydot_and_graphviz()
    import pydot
    graph = pydot.Dot(rankdir='LR')
    nodes, edges = _get_nodes_and_edges(dag)
    name_generator = _DAGNodeNameGenerator()
    node_names = {}
    for node in nodes:
        node_names[node] = name_generator.get_node_name(node)
    for edge in edges:
        graph.add_edge(pydot.Edge(node_names[edge[0]], node_names[edge[1]]))
    if len(nodes) == 1 and len(edges) == 0:
        graph.add_node(pydot.Node(node_names[nodes[0]]))
    return graph