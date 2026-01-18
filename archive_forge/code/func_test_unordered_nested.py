from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_unordered_nested(self):
    a, b, c, d = test_utils.make_many(4)
    flo = uf.Flow('test')
    flo.add(a, b)
    flo2 = lf.Flow('test2')
    flo2.add(c, d)
    flo.add(flo2)
    g = _replicate_graph_with_names(compiler.PatternCompiler(flo).compile())
    self.assertEqual(8, len(g))
    self.assertCountEqual(g.edges(), [('test', 'a'), ('test', 'b'), ('test', 'test2'), ('test2', 'c'), ('c', 'd'), ('d', 'test2[$]'), ('test2[$]', 'test[$]'), ('a', 'test[$]'), ('b', 'test[$]')])