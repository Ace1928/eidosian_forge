from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def _replicate_graph_with_names(compilation):
    g = compilation.execution_graph
    n_g = g.__class__(name=g.name)
    for node, node_data in g.nodes(data=True):
        n_g.add_node(node.name, attr_dict=node_data)
    for u, v, u_v_data in g.edges(data=True):
        n_g.add_edge(u.name, v.name, attr_dict=u_v_data)
    return n_g