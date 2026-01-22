import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
class SuspendTest(utils.EngineTestBase):

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

    def test_suspend_and_resume_linear_flow_on_revert(self):
        flow = lf.Flow('linear').add(utils.ProgressingTask('a'), utils.ProgressingTask('b'), utils.FailingTask('c'))
        engine = self._make_engine(flow)
        with SuspendingListener(engine, task_name='b', task_state=states.REVERTED) as capturer:
            engine.run()
        expected = ['a.t RUNNING', 'a.t SUCCESS(5)', 'b.t RUNNING', 'b.t SUCCESS(5)', 'c.t RUNNING', 'c.t FAILURE(Failure: RuntimeError: Woot!)', 'c.t REVERTING', 'c.t REVERTED(None)', 'b.t REVERTING', 'b.t REVERTED(None)']
        self.assertEqual(expected, capturer.values)
        engine2 = self._make_engine(flow, engine.storage._flowdetail)
        with utils.CaptureListener(engine2, capture_flow=False) as capturer2:
            self.assertRaisesRegex(RuntimeError, '^Woot', engine2.run)
        self.assertEqual(states.REVERTED, engine2.storage.get_flow_state())
        expected = ['a.t REVERTING', 'a.t REVERTED(None)']
        self.assertEqual(expected, capturer2.values)

    def test_suspend_and_revert_even_if_task_is_gone(self):
        flow = lf.Flow('linear').add(utils.ProgressingTask('a'), utils.ProgressingTask('b'), utils.FailingTask('c'))
        engine = self._make_engine(flow)
        with SuspendingListener(engine, task_name='b', task_state=states.REVERTED) as capturer:
            engine.run()
        expected = ['a.t RUNNING', 'a.t SUCCESS(5)', 'b.t RUNNING', 'b.t SUCCESS(5)', 'c.t RUNNING', 'c.t FAILURE(Failure: RuntimeError: Woot!)', 'c.t REVERTING', 'c.t REVERTED(None)', 'b.t REVERTING', 'b.t REVERTED(None)']
        self.assertEqual(expected, capturer.values)
        flow2 = lf.Flow('linear').add(utils.ProgressingTask('a'), utils.ProgressingTask('b'))
        engine2 = self._make_engine(flow2, engine.storage._flowdetail)
        with utils.CaptureListener(engine2, capture_flow=False) as capturer2:
            self.assertRaisesRegex(RuntimeError, '^Woot', engine2.run)
        self.assertEqual(states.REVERTED, engine2.storage.get_flow_state())
        expected = ['a.t REVERTING', 'a.t REVERTED(None)']
        self.assertEqual(expected, capturer2.values)

    def test_storage_is_rechecked(self):
        flow = lf.Flow('linear').add(utils.ProgressingTask('b', requires=['foo']), utils.ProgressingTask('c'))
        engine = self._make_engine(flow)
        engine.storage.inject({'foo': 'bar'})
        with SuspendingListener(engine, task_name='b', task_state=states.SUCCESS):
            engine.run()
        self.assertEqual(states.SUSPENDED, engine.storage.get_flow_state())
        engine.storage.save(engine.storage.injector_name, {}, states.SUCCESS)
        self.assertRaises(exc.MissingDependencies, engine.run)