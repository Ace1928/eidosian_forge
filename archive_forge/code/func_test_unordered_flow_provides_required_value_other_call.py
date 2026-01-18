from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_provides_required_value_other_call(self):
    flow = uf.Flow('uf')
    flow.add(utils.TaskOneArg('task2'))
    flow.add(utils.TaskOneReturn('task1', provides='x'))
    self.assertEqual(2, len(flow))
    self.assertEqual(set(['x']), flow.provides)
    self.assertEqual(set(['x']), flow.requires)