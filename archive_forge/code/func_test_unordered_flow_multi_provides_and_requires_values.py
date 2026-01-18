from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_multi_provides_and_requires_values(self):
    flow = uf.Flow('uf').add(utils.TaskMultiArgMultiReturn('task1', rebind=['a', 'b', 'c'], provides=['d', 'e', 'f']), utils.TaskMultiArgMultiReturn('task2', provides=['i', 'j', 'k']))
    self.assertEqual(set(['a', 'b', 'c', 'x', 'y', 'z']), flow.requires)
    self.assertEqual(set(['d', 'e', 'f', 'i', 'j', 'k']), flow.provides)