from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_unordered(self):
    a, b, c, d = test_utils.make_many(4)
    flo = uf.Flow('test')
    flo.add(a, b, c, d)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(6, len(g))
    self.assertCountEqual(g.edges(), [('test', 'a'), ('test', 'b'), ('test', 'c'), ('test', 'd'), ('a', 'test[$]'), ('b', 'test[$]'), ('c', 'test[$]'), ('d', 'test[$]')])
    self.assertEqual(set(['test']), set(g.no_predecessors_iter()))