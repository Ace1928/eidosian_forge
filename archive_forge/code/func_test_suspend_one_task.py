import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_suspend_one_task(self):
    flow = utils.ProgressingTask('a')
    engine = self._make_engine(flow)
    with SuspendingListener(engine, task_name='b', task_state=states.SUCCESS) as capturer:
        engine.run()
    self.assertEqual(states.SUCCESS, engine.storage.get_flow_state())
    expected = ['a.t RUNNING', 'a.t SUCCESS(5)']
    self.assertEqual(expected, capturer.values)
    with SuspendingListener(engine, task_name='b', task_state=states.SUCCESS) as capturer:
        engine.run()
    self.assertEqual(states.SUCCESS, engine.storage.get_flow_state())
    expected = []
    self.assertEqual(expected, capturer.values)