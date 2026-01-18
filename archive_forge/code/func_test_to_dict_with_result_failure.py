from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_to_dict_with_result_failure(self):
    a_failure = failure.Failure.from_exception(RuntimeError('Woot!'))
    expected = self.request_to_dict(result=('failure', a_failure.to_dict()))
    request = self.request(result=a_failure)
    self.assertEqual(expected, request.to_dict())