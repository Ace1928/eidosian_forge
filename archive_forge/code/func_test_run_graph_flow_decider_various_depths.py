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
def test_run_graph_flow_decider_various_depths(self):
    sub_flow_1 = gf.Flow('g_1')
    g_1_1 = utils.ProgressingTask(name='g_1-1')
    sub_flow_1.add(g_1_1)
    g_1 = utils.ProgressingTask(name='g-1')
    g_2 = utils.ProgressingTask(name='g-2')
    g_3 = utils.ProgressingTask(name='g-3')
    g_4 = utils.ProgressingTask(name='g-4')
    for a_depth, ran_how_many in [('all', 1), ('atom', 4), ('flow', 2), ('neighbors', 3)]:
        flow = gf.Flow('g')
        flow.add(g_1, g_2, sub_flow_1, g_3, g_4)
        flow.link(g_1, g_2, decider=lambda history: False, decider_depth=a_depth)
        flow.link(g_2, sub_flow_1)
        flow.link(g_2, g_3)
        flow.link(g_3, g_4)
        flow.link(g_1, sub_flow_1, decider=lambda history: True, decider_depth=a_depth)
        e = self._make_engine(flow)
        with utils.CaptureListener(e, capture_flow=False) as capturer:
            e.run()
        ran_tasks = 0
        for outcome in capturer.values:
            if outcome.endswith('RUNNING'):
                ran_tasks += 1
        self.assertEqual(ran_how_many, ran_tasks)