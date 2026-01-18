import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.utils import eventlet_utils as eu
def test_resume_flow_that_had_been_interrupted_during_retrying(self):
    flow = lf.Flow('flow-1', retry.Times(3, 'r1')).add(utils.ProgressingTask('t1'), utils.ProgressingTask('t2'), utils.ProgressingTask('t3'))
    engine = self._make_engine(flow)
    engine.compile()
    engine.prepare()
    with utils.CaptureListener(engine) as capturer:
        engine.storage.set_atom_state('r1', st.RETRYING)
        engine.storage.set_atom_state('t1', st.PENDING)
        engine.storage.set_atom_state('t2', st.REVERTED)
        engine.storage.set_atom_state('t3', st.REVERTED)
        engine.run()
    expected = ['flow-1.f RUNNING', 't2.t PENDING', 't3.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(1)', 't1.t RUNNING', 't1.t SUCCESS(5)', 't2.t RUNNING', 't2.t SUCCESS(5)', 't3.t RUNNING', 't3.t SUCCESS(5)', 'flow-1.f SUCCESS']
    self.assertEqual(expected, capturer.values)