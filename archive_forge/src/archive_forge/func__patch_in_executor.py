from taskflow.engines.worker_based import engine
from taskflow.engines.worker_based import executor
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import backends
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.utils import persistence_utils as pu
def _patch_in_executor(self):
    executor_mock, executor_inst_mock = self.patchClass(engine.executor, 'WorkerTaskExecutor', attach_as='executor')
    return (executor_mock, executor_inst_mock)