from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_retry_subflows_hierarchy(self):
    c1 = retry.AlwaysRevert('c1')
    a, b, c, d = test_utils.make_many(4)
    inner_flo = lf.Flow('test2').add(b, c)
    flo = lf.Flow('test', c1).add(a, inner_flo, d)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(9, len(g))
    self.assertCountEqual(g.edges(data=True), [('test', 'c1', {'invariant': True}), ('c1', 'a', {'invariant': True, 'retry': True}), ('a', 'test2', {'invariant': True}), ('test2', 'b', {'invariant': True}), ('b', 'c', {'invariant': True}), ('c', 'test2[$]', {'invariant': True}), ('test2[$]', 'd', {'invariant': True}), ('d', 'test[$]', {'invariant': True})])
    self.assertIs(c1, g.nodes['a']['retry'])
    self.assertIs(c1, g.nodes['d']['retry'])
    self.assertIs(c1, g.nodes['b']['retry'])
    self.assertIs(c1, g.nodes['c']['retry'])
    self.assertIsNone(g.nodes['c1'].get('retry'))