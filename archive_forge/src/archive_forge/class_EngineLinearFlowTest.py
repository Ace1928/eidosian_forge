import collections
import contextlib
import functools
import threading
import futurist
import testtools
import taskflow.engines
from taskflow.engines.action_engine import engine as eng
from taskflow.engines.worker_based import engine as w_eng
from taskflow.engines.worker_based import worker as wkr
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow.persistence import models
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.types import graph as gr
from taskflow.utils import eventlet_utils as eu
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils as tu
class EngineLinearFlowTest(utils.EngineTestBase):

    def test_run_empty_linear_flow(self):
        flow = lf.Flow('flow-1')
        engine = self._make_engine(flow)
        self.assertEqual(_EMPTY_TRANSITIONS, list(engine.run_iter()))

    def test_overlap_parent_sibling_expected_result(self):
        flow = lf.Flow('flow-1')
        flow.add(utils.ProgressingTask(provides='source'))
        flow.add(utils.TaskOneReturn(provides='source'))
        subflow = lf.Flow('flow-2')
        subflow.add(utils.AddOne())
        flow.add(subflow)
        engine = self._make_engine(flow)
        engine.run()
        results = engine.storage.fetch_all()
        self.assertEqual(2, results['result'])

    def test_overlap_parent_expected_result(self):
        flow = lf.Flow('flow-1')
        flow.add(utils.ProgressingTask(provides='source'))
        subflow = lf.Flow('flow-2')
        subflow.add(utils.TaskOneReturn(provides='source'))
        subflow.add(utils.AddOne())
        flow.add(subflow)
        engine = self._make_engine(flow)
        engine.run()
        results = engine.storage.fetch_all()
        self.assertEqual(2, results['result'])

    def test_overlap_sibling_expected_result(self):
        flow = lf.Flow('flow-1')
        flow.add(utils.ProgressingTask(provides='source'))
        flow.add(utils.TaskOneReturn(provides='source'))
        flow.add(utils.AddOne())
        engine = self._make_engine(flow)
        engine.run()
        results = engine.storage.fetch_all()
        self.assertEqual(2, results['result'])

    def test_sequential_flow_interrupted_externally(self):
        flow = lf.Flow('flow-1').add(utils.ProgressingTask(name='task1'), utils.ProgressingTask(name='task2'), utils.ProgressingTask(name='task3'))
        engine = self._make_engine(flow)

        def _run_engine_and_raise():
            engine_states = {}
            engine_it = engine.run_iter()
            while True:
                try:
                    engine_state = next(engine_it)
                    if engine_state not in engine_states:
                        engine_states[engine_state] = 1
                    else:
                        engine_states[engine_state] += 1
                    if engine_states.get(states.SCHEDULING) == 2:
                        engine_state = engine_it.throw(IOError('I Broke'))
                        if engine_state not in engine_states:
                            engine_states[engine_state] = 1
                        else:
                            engine_states[engine_state] += 1
                except StopIteration:
                    break
        self.assertRaises(IOError, _run_engine_and_raise)
        self.assertEqual(states.FAILURE, engine.storage.get_flow_state())

    def test_sequential_flow_one_task(self):
        flow = lf.Flow('flow-1').add(utils.ProgressingTask(name='task1'))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)

    def test_sequential_flow_two_tasks(self):
        flow = lf.Flow('flow-2').add(utils.ProgressingTask(name='task1'), utils.ProgressingTask(name='task2'))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)
        self.assertEqual(2, len(flow))

    def test_sequential_flow_two_tasks_iter(self):
        flow = lf.Flow('flow-2').add(utils.ProgressingTask(name='task1'), utils.ProgressingTask(name='task2'))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            gathered_states = list(engine.run_iter())
        self.assertTrue(len(gathered_states) > 0)
        expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)
        self.assertEqual(2, len(flow))

    def test_sequential_flow_iter_suspend_resume(self):
        flow = lf.Flow('flow-2').add(utils.ProgressingTask(name='task1'), utils.ProgressingTask(name='task2'))
        lb, fd = p_utils.temporary_flow_detail(self.backend)
        engine = self._make_engine(flow, flow_detail=fd)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            it = engine.run_iter()
            gathered_states = []
            suspend_it = None
            while True:
                try:
                    s = it.send(suspend_it)
                    gathered_states.append(s)
                    if s == states.WAITING:
                        suspend_it = True
                except StopIteration:
                    break
        self.assertTrue(len(gathered_states) > 0)
        expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)
        self.assertEqual(states.SUSPENDED, engine.storage.get_flow_state())
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            gathered_states = list(engine.run_iter())
        self.assertTrue(len(gathered_states) > 0)
        expected = ['task2.t RUNNING', 'task2.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)
        self.assertEqual(states.SUCCESS, engine.storage.get_flow_state())

    def test_revert_removes_data(self):
        flow = lf.Flow('revert-removes').add(utils.TaskOneReturn(provides='one'), utils.TaskMultiReturn(provides=('a', 'b', 'c')), utils.FailingTask(name='fail'))
        engine = self._make_engine(flow)
        self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        self.assertEqual({}, engine.storage.fetch_all())

    def test_revert_provided(self):
        flow = lf.Flow('revert').add(utils.GiveBackRevert('giver'), utils.FailingTask(name='fail'))
        engine = self._make_engine(flow, store={'value': 0})
        self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        self.assertEqual(2, engine.storage.get_revert_result('giver'))

    def test_nasty_revert(self):
        flow = lf.Flow('revert').add(utils.NastyTask('nasty'), utils.FailingTask(name='fail'))
        engine = self._make_engine(flow)
        self.assertFailuresRegexp(RuntimeError, '^Gotcha', engine.run)
        fail = engine.storage.get_revert_result('nasty')
        self.assertIsNotNone(fail.check(RuntimeError))
        exec_failures = engine.storage.get_execute_failures()
        self.assertIn('fail', exec_failures)
        rev_failures = engine.storage.get_revert_failures()
        self.assertIn('nasty', rev_failures)

    def test_sequential_flow_nested_blocks(self):
        flow = lf.Flow('nested-1').add(utils.ProgressingTask('task1'), lf.Flow('inner-1').add(utils.ProgressingTask('task2')))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            engine.run()
        expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)']
        self.assertEqual(expected, capturer.values)

    def test_revert_exception_is_reraised(self):
        flow = lf.Flow('revert-1').add(utils.NastyTask(), utils.FailingTask(name='fail'))
        engine = self._make_engine(flow)
        self.assertFailuresRegexp(RuntimeError, '^Gotcha', engine.run)

    def test_revert_not_run_task_is_not_reverted(self):
        flow = lf.Flow('revert-not-run').add(utils.FailingTask('fail'), utils.NeverRunningTask())
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        expected = ['fail.t RUNNING', 'fail.t FAILURE(Failure: RuntimeError: Woot!)', 'fail.t REVERTING', 'fail.t REVERTED(None)']
        self.assertEqual(expected, capturer.values)

    def test_correctly_reverts_children(self):
        flow = lf.Flow('root-1').add(utils.ProgressingTask('task1'), lf.Flow('child-1').add(utils.ProgressingTask('task2'), utils.FailingTask('fail')))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)', 'fail.t RUNNING', 'fail.t FAILURE(Failure: RuntimeError: Woot!)', 'fail.t REVERTING', 'fail.t REVERTED(None)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'task1.t REVERTING', 'task1.t REVERTED(None)']
        self.assertEqual(expected, capturer.values)