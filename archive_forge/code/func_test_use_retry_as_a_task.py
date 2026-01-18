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
def test_use_retry_as_a_task(self):
    flow = lf.Flow('test').add(utils.OneReturnRetry(provides='x'))
    engine = self._make_engine(flow)
    self.assertRaises(TypeError, engine.run)