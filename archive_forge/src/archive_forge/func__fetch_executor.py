import futurist
from futurist import waiters
from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor as base_executor
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import executor as worker_executor
from taskflow.engines.worker_based import server as worker_server
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import threading_utils
def _fetch_executor(self):
    executor = worker_executor.WorkerTaskExecutor(uuidutils.generate_uuid(), TEST_EXCHANGE, [TEST_TOPIC], transport='memory', transport_options={'polling_interval': POLLING_INTERVAL})
    return executor