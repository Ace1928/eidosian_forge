from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
class LinearScopingTest(test.TestCase):

    def test_unknown(self):
        r = lf.Flow('root')
        r_1 = test_utils.TaskOneReturn('root.1')
        r.add(r_1)
        r_2 = test_utils.TaskOneReturn('root.2')
        c = compiler.PatternCompiler(r).compile()
        self.assertRaises(ValueError, _get_scopes, c, r_2)

    def test_empty(self):
        r = lf.Flow('root')
        r_1 = test_utils.TaskOneReturn('root.1')
        r.add(r_1)
        c = compiler.PatternCompiler(r).compile()
        self.assertIn(r_1, c.execution_graph)
        self.assertIsNotNone(c.hierarchy.find(r_1))
        walker = sc.ScopeWalker(c, r_1)
        scopes = list(walker)
        self.assertEqual([], scopes)

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

    def test_nested_prior_linear(self):
        r = lf.Flow('root')
        r.add(test_utils.TaskOneReturn('root.1'), test_utils.TaskOneReturn('root.2'))
        sub_r = lf.Flow('subroot')
        sub_r_1 = test_utils.TaskOneReturn('subroot.1')
        sub_r.add(sub_r_1)
        r.add(sub_r)
        c = compiler.PatternCompiler(r).compile()
        self.assertEqual([[], ['root.2', 'root.1']], _get_scopes(c, sub_r_1))

    def test_nested_prior_linear_begin_middle_end(self):
        r = lf.Flow('root')
        begin_r = test_utils.TaskOneReturn('root.1')
        r.add(begin_r, test_utils.TaskOneReturn('root.2'))
        middle_r = test_utils.TaskOneReturn('root.3')
        r.add(middle_r)
        sub_r = lf.Flow('subroot')
        sub_r.add(test_utils.TaskOneReturn('subroot.1'), test_utils.TaskOneReturn('subroot.2'))
        r.add(sub_r)
        end_r = test_utils.TaskOneReturn('root.4')
        r.add(end_r)
        c = compiler.PatternCompiler(r).compile()
        self.assertEqual([], _get_scopes(c, begin_r))
        self.assertEqual([['root.2', 'root.1']], _get_scopes(c, middle_r))
        self.assertEqual([['subroot.2', 'subroot.1', 'root.3', 'root.2', 'root.1']], _get_scopes(c, end_r))