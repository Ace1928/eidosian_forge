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