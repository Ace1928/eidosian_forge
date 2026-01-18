from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_retry_in_nested_flows(self):
    c1 = retry.AlwaysRevert('c1')
    c2 = retry.AlwaysRevert('c2')
    inner_flo = lf.Flow('test2', c2)
    flo = lf.Flow('test', c1).add(inner_flo)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(6, len(g))
    self.assertCountEqual(g.edges(data=True), [('test', 'c1', {'invariant': True}), ('c1', 'test2', {'invariant': True, 'retry': True}), ('test2', 'c2', {'invariant': True}), ('c2', 'test2[$]', {'invariant': True}), ('test2[$]', 'test[$]', {'invariant': True})])
    self.assertIs(c1, g.nodes['c2']['retry'])
    self.assertCountEqual(['test'], list(g.no_predecessors_iter()))
    self.assertCountEqual(['test[$]'], list(g.no_successors_iter()))