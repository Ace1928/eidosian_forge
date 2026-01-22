import contextlib
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import task
from taskflow import test
from taskflow.utils import persistence_utils as p_utils
def _make_engine(self, flow, flow_detail=None, backend=None):
    e = taskflow.engines.load(flow, flow_detail=flow_detail, backend=backend)
    e.compile()
    e.prepare()
    return e