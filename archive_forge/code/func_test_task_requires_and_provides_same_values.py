from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_task_requires_and_provides_same_values(self):
    flow = lf.Flow('lf', utils.TaskOneArgOneReturn('rt', requires='x', provides='x'))
    self.assertEqual(set('x'), flow.requires)
    self.assertEqual(set('x'), flow.provides)