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
def test_run_graph_flow_decider_revert(self):
    flow = gf.Flow('g')
    a = utils.NoopTask('a')
    b = utils.NoopTask('b')
    c = utils.FailingTask('c')
    flow.add(a, b, c)
    flow.link(a, b, decider=lambda history: False, decider_depth='atom')
    flow.link(b, c)
    e = self._make_engine(flow)
    with utils.CaptureListener(e, capture_flow=False) as capturer:
        self.assertRaises((RuntimeError, exc.WrappedFailure), e.run)
    expected = ['a.t RUNNING', 'a.t SUCCESS(None)', 'b.t IGNORE', 'c.t RUNNING', 'c.t FAILURE(Failure: RuntimeError: Woot!)', 'c.t REVERTING', 'c.t REVERTED(None)', 'a.t REVERTING', 'a.t REVERTED(None)']
    self.assertEqual(expected, capturer.values)