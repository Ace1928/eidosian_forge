import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.utils import eventlet_utils as eu
def test_retry_fails(self):
    r = FailingRetry()
    flow = lf.Flow('testflow', r)
    engine = self._make_engine(flow)
    self.assertRaisesRegex(ValueError, '^OMG', engine.run)
    self.assertEqual(1, len(engine.storage.get_retry_histories()))
    self.assertEqual(0, len(r.history))
    self.assertEqual([], list(r.history.outcomes_iter()))
    self.assertIsNotNone(r.history.failure)
    self.assertTrue(r.history.caused_by(ValueError, include_retry=True))