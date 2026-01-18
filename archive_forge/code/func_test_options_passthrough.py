import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
def test_options_passthrough(self):
    f = self._make_dummy_flow()
    e = taskflow.engines.load(f, pass_1=1, pass_2=2)
    self.assertEqual({'pass_1': 1, 'pass_2': 2}, e.options)