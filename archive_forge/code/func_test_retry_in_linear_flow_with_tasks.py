from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_retry_in_linear_flow_with_tasks(self):
    c = retry.AlwaysRevert('c')
    a, b = test_utils.make_many(2)
    flo = lf.Flow('test', c).add(a, b)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(5, len(g))
    self.assertCountEqual(g.edges(data=True), [('test', 'c', {'invariant': True}), ('a', 'b', {'invariant': True}), ('c', 'a', {'invariant': True, 'retry': True}), ('b', 'test[$]', {'invariant': True})])
    self.assertCountEqual(['test'], g.no_predecessors_iter())
    self.assertCountEqual(['test[$]'], g.no_successors_iter())
    self.assertIs(c, g.nodes['a']['retry'])
    self.assertIs(c, g.nodes['b']['retry'])