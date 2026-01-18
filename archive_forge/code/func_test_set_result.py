from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_set_result(self):
    request = self.request()
    request.set_result(111)
    result = request.future.result()
    self.assertEqual((executor.EXECUTED, 111), result)