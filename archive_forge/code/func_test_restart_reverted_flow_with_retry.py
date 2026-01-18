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
def test_restart_reverted_flow_with_retry(self):
    flow = lf.Flow('test', retry=utils.OneReturnRetry(provides='x')).add(utils.FailingTask('fail'))
    engine = self._make_engine(flow)
    self.assertRaisesRegex(RuntimeError, '^Woot', engine.run)
    self.assertRaisesRegex(RuntimeError, '^Woot', engine.run)