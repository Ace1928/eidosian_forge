from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
def test_single_prior_linear(self):
    r = lf.Flow('root')
    r_1 = test_utils.TaskOneReturn('root.1')
    r_2 = test_utils.TaskOneReturn('root.2')
    r.add(r_1, r_2)
    c = compiler.PatternCompiler(r).compile()
    for a in r:
        self.assertIn(a, c.execution_graph)
        self.assertIsNotNone(c.hierarchy.find(a))
    self.assertEqual([], _get_scopes(c, r_1))
    self.assertEqual([['root.1']], _get_scopes(c, r_2))