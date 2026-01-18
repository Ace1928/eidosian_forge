import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_save_all_in_one(self):
    flow = utils.TaskMultiReturn(provides='all_data')
    engine = self._make_engine(flow)
    engine.run()
    self.assertEqual({'all_data': (1, 3, 5)}, engine.storage.fetch_all())