import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_partial_arguments_mapping(self):
    flow = utils.TaskMultiArgOneReturn(provides='result', rebind={'x': 'a'})
    engine = self._make_engine(flow)
    engine.storage.inject({'x': 1, 'y': 4, 'z': 9, 'a': 17})
    engine.run()
    self.assertEqual({'x': 1, 'y': 4, 'z': 9, 'a': 17, 'result': 30}, engine.storage.fetch_all())