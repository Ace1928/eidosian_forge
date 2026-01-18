from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_task_without_dependencies(self):
    flow = utils.TaskNoRequiresNoReturns()
    self.assertEqual(set(), flow.requires)
    self.assertEqual(set(), flow.provides)