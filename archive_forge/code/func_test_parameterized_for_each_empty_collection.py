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
def test_parameterized_for_each_empty_collection(self):
    values = []
    retry1 = retry.ParameterizedForEach('r1', provides='x')
    flow = lf.Flow('flow-1', retry1).add(utils.ConditionalTask('t1'))
    engine = self._make_engine(flow)
    engine.storage.inject({'values': values, 'y': 1})
    self.assertRaisesRegex(exc.NotFound, '^No elements left', engine.run)