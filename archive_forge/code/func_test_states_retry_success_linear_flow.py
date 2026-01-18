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
def test_states_retry_success_linear_flow(self):
    flow = lf.Flow('flow-1', retry.Times(4, 'r1', provides='x')).add(utils.ProgressingTask('task1'), utils.ConditionalTask('task2'))
    engine = self._make_engine(flow)
    engine.storage.inject({'y': 2})
    with utils.CaptureListener(engine) as capturer:
        engine.run()
    self.assertEqual({'y': 2, 'x': 2}, engine.storage.fetch_all())
    expected = ['flow-1.f RUNNING', 'r1.r RUNNING', 'r1.r SUCCESS(1)', 'task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t FAILURE(Failure: RuntimeError: Woot!)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'task1.t REVERTING', 'task1.t REVERTED(None)', 'r1.r RETRYING', 'task1.t PENDING', 'task2.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(2)', 'task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(None)', 'flow-1.f SUCCESS']
    self.assertEqual(expected, capturer.values)