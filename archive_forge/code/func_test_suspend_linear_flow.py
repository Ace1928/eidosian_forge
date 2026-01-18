import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_suspend_linear_flow(self):
    flow = lf.Flow('linear').add(utils.ProgressingTask('a'), utils.ProgressingTask('b'), utils.ProgressingTask('c'))
    engine = self._make_engine(flow)
    with SuspendingListener(engine, task_name='b', task_state=states.SUCCESS) as capturer:
        engine.run()
    self.assertEqual(states.SUSPENDED, engine.storage.get_flow_state())
    expected = ['a.t RUNNING', 'a.t SUCCESS(5)', 'b.t RUNNING', 'b.t SUCCESS(5)']
    self.assertEqual(expected, capturer.values)
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        engine.run()
    self.assertEqual(states.SUCCESS, engine.storage.get_flow_state())
    expected = ['c.t RUNNING', 'c.t SUCCESS(5)']
    self.assertEqual(expected, capturer.values)