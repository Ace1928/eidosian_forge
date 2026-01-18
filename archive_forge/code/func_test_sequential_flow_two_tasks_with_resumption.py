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
def test_sequential_flow_two_tasks_with_resumption(self):
    flow = lf.Flow('lf-2-r').add(utils.ProgressingTask(name='task1', provides='x1'), utils.ProgressingTask(name='task2', provides='x2'))
    lb, fd = p_utils.temporary_flow_detail(self.backend)
    td = models.TaskDetail(name='task1', uuid='42')
    td.state = states.SUCCESS
    td.results = 17
    fd.add(td)
    with contextlib.closing(self.backend.get_connection()) as conn:
        fd.update(conn.update_flow_details(fd))
        td.update(conn.update_atom_details(td))
    engine = self._make_engine(flow, fd)
    with utils.CaptureListener(engine, capture_flow=False) as capturer:
        engine.run()
    expected = ['task2.t RUNNING', 'task2.t SUCCESS(5)']
    self.assertEqual(expected, capturer.values)
    self.assertEqual({'x1': 17, 'x2': 5}, engine.storage.fetch_all())