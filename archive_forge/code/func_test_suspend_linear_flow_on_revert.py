import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_suspend_linear_flow_on_revert(self):
    flow = lf.Flow('linear').add(utils.ProgressingTask('a'), utils.ProgressingTask('b'), utils.FailingTask('c'))
    engine = self._make_engine(flow)
    with SuspendingListener(engine, task_name='b', task_state=states.REVERTED) as capturer:
        engine.run()
    self.assertEqual(states.SUSPENDED, engine.storage.get_flow_state())
    expected = ['a.t RUNNING', 'a.t SUCCESS(5)', 'b.t RUNNING', 'b.t SUCCESS(5)', 'c.t RUNNING', 'c.t FAILURE(Failure: RuntimeError: Woot!)', 'c.t REVERTING', 'c.t REVERTED(None)', 'b.t REVERTING', 'b.t REVERTED(None)']
    self.assertEqual(expected, capturer.values)
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        self.assertRaisesRegex(RuntimeError, '^Woot', engine.run)
    self.assertEqual(states.REVERTED, engine.storage.get_flow_state())
    expected = ['a.t REVERTING', 'a.t REVERTED(None)']
    self.assertEqual(expected, capturer.values)