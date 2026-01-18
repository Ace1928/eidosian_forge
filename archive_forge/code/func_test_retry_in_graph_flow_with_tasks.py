from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_retry_in_graph_flow_with_tasks(self):
    r = retry.AlwaysRevert('r')
    a, b, c = test_utils.make_many(3)
    flo = gf.Flow('test', r).add(a, b, c).link(b, c)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertCountEqual(g.edges(data=True), [('test', 'r', {'invariant': True}), ('r', 'a', {'invariant': True, 'retry': True}), ('r', 'b', {'invariant': True, 'retry': True}), ('b', 'c', {'manual': True}), ('a', 'test[$]', {'invariant': True}), ('c', 'test[$]', {'invariant': True})])
    self.assertCountEqual(['test'], g.no_predecessors_iter())
    self.assertCountEqual(['test[$]'], g.no_successors_iter())
    self.assertIs(r, g.nodes['a']['retry'])
    self.assertIs(r, g.nodes['b']['retry'])
    self.assertIs(r, g.nodes['c']['retry'])