import futurist
import testtools
from taskflow.engines.action_engine import engine
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import process_executor
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
from taskflow.utils import persistence_utils as pu
def test_sync_executor_creation(self):
    with futurist.SynchronousExecutor() as e:
        eng = self._create_engine(executor=e)
        self.assertIsInstance(eng._task_executor, executor.ParallelThreadTaskExecutor)