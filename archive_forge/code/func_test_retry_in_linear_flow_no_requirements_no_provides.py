from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_retry_in_linear_flow_no_requirements_no_provides(self):
    flow = lf.Flow('lf', retry.AlwaysRevert('rt'))
    self.assertEqual(set(), flow.requires)
    self.assertEqual(set(), flow.provides)