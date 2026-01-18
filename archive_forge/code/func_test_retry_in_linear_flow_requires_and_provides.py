from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_retry_in_linear_flow_requires_and_provides(self):
    flow = lf.Flow('lf', retry.AlwaysRevert('rt', requires=['x', 'y'], provides=['a', 'b']))
    self.assertEqual(set(['x', 'y']), flow.requires)
    self.assertEqual(set(['a', 'b']), flow.provides)