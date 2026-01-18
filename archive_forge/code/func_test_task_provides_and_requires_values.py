from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_task_provides_and_requires_values(self):
    flow = utils.TaskMultiArgMultiReturn(provides=['a', 'b', 'c'])
    self.assertEqual(set(['x', 'y', 'z']), flow.requires)
    self.assertEqual(set(['a', 'b', 'c']), flow.provides)