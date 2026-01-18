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
def test_sequential_flow_two_tasks_iter(self):
    flow = lf.Flow('flow-2').add(utils.ProgressingTask(name='task1'), utils.ProgressingTask(name='task2'))
    engine = self._make_engine(flow)
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        gathered_states = list(engine.run_iter())
    self.assertTrue(len(gathered_states) > 0)
    expected = ['task1.t RUNNING', 'task1.t SUCCESS(5)', 'task2.t RUNNING', 'task2.t SUCCESS(5)']
    self.assertEqual(expected, capturer.values)
    self.assertEqual(2, len(flow))