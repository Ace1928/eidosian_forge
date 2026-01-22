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
class EngineLinearAndUnorderedExceptionsTest(utils.EngineTestBase):

    def test_revert_ok_for_unordered_in_linear(self):
        flow = lf.Flow('p-root').add(utils.ProgressingTask(name='task1'), utils.ProgressingTask(name='task2'), uf.Flow('p-inner').add(utils.ProgressingTask(name='task3'), utils.FailingTask('fail')))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        possible_values_no_task3 = ['task1.t RUNNING', 'task2.t RUNNING', 'fail.t FAILURE(Failure: RuntimeError: Woot!)', 'task2.t REVERTED(None)', 'task1.t REVERTED(None)']
        self.assertIsSuperAndSubsequence(capturer.values, possible_values_no_task3)
        if 'task3' in capturer.values:
            possible_values_task3 = ['task1.t RUNNING', 'task2.t RUNNING', 'task3.t RUNNING', 'task3.t REVERTED(None)', 'task2.t REVERTED(None)', 'task1.t REVERTED(None)']
            self.assertIsSuperAndSubsequence(capturer.values, possible_values_task3)

    def test_revert_raises_for_unordered_in_linear(self):
        flow = lf.Flow('p-root').add(utils.ProgressingTask(name='task1'), utils.ProgressingTask(name='task2'), uf.Flow('p-inner').add(utils.ProgressingTask(name='task3'), utils.NastyFailingTask(name='nasty')))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False, skip_tasks=['nasty']) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Gotcha', engine.run)
        possible_values = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)', 'task3.t RUNNING', 'task3.t SUCCESS(5)', 'task3.t REVERTING', 'task3.t REVERTED(None)']
        self.assertIsSuperAndSubsequence(possible_values, capturer.values)
        possible_values_no_task3 = ['task1.t RUNNING', 'task2.t RUNNING']
        self.assertIsSuperAndSubsequence(capturer.values, possible_values_no_task3)

    def test_revert_ok_for_linear_in_unordered(self):
        flow = uf.Flow('p-root').add(utils.ProgressingTask(name='task1'), lf.Flow('p-inner').add(utils.ProgressingTask(name='task2'), utils.FailingTask('fail')))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Woot', engine.run)
        self.assertIn('fail.t FAILURE(Failure: RuntimeError: Woot!)', capturer.values)
        if 'task1' in capturer.values:
            task1_story = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task1.t REVERTED(None)']
            self.assertIsSuperAndSubsequence(capturer.values, task1_story)
        task2_story = ['task2.t RUNNING', 'task2.t SUCCESS(5)', 'task2.t REVERTED(None)']
        self.assertIsSuperAndSubsequence(capturer.values, task2_story)

    def test_revert_raises_for_linear_in_unordered(self):
        flow = uf.Flow('p-root').add(utils.ProgressingTask(name='task1'), lf.Flow('p-inner').add(utils.ProgressingTask(name='task2'), utils.NastyFailingTask()))
        engine = self._make_engine(flow)
        with utils.CaptureListener(engine, capture_flow=False) as capturer:
            self.assertFailuresRegexp(RuntimeError, '^Gotcha', engine.run)
        self.assertNotIn('task2.t REVERTED(None)', capturer.values)