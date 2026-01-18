from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_retry_two_tasks_provide_same_value(self):
    flow = uf.Flow('uf', retry.AlwaysRevert('rt', provides=['y']))
    flow.add(utils.TaskOneReturn('t1', provides=['x']), utils.TaskOneReturn('t2', provides=['x']))
    self.assertEqual(set(['x', 'y']), flow.provides)