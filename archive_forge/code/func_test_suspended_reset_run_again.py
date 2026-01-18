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
def test_suspended_reset_run_again(self):
    task1 = utils.ProgressingTask(name='task1')
    task2 = utils.ProgressingTask(name='task2')
    task3 = utils.ProgressingTask(name='task3')
    flow = lf.Flow('root')
    flow.add(task1, task2, task3)
    engine = self._make_engine(flow)
    suspend_at = object()
    expected_states = [states.RESUMING, states.SCHEDULING, states.WAITING, states.ANALYZING, states.SCHEDULING, states.WAITING, suspend_at, states.SUSPENDED]
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        for i, st in enumerate(engine.run_iter()):
            expected = expected_states[i]
            if expected is suspend_at:
                engine.suspend()
            else:
                self.assertEqual(expected, st)
    expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)']
    self.assertEqual(expected, capturer.values)
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        engine.run()
    expected = ['task3.t RUNNING', 'task3.t SUCCESS(5)']
    self.assertEqual(expected, capturer.values)
    engine.reset()
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        engine.run()
    expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)', 'task3.t RUNNING', 'task3.t SUCCESS(5)']
    self.assertEqual(expected, capturer.values)