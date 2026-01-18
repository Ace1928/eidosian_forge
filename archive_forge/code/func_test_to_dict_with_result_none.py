from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_to_dict_with_result_none(self):
    request = self.request(result=None)
    self.assertEqual(self.request_to_dict(result=('success', None)), request.to_dict())