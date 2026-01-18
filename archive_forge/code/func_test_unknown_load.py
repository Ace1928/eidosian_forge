import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
def test_unknown_load(self):
    f = self._make_dummy_flow()
    self.assertRaises(exc.NotFound, taskflow.engines.load, f, engine='not_really_any_engine')