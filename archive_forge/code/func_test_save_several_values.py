import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_save_several_values(self):
    flow = utils.TaskMultiReturn(provides=('badger', 'mushroom', 'snake'))
    engine = self._make_engine(flow)
    engine.run()
    self.assertEqual({'badger': 1, 'mushroom': 3, 'snake': 5}, engine.storage.fetch_all())