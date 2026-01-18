from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
@mock.patch('oslo_utils.timeutils.now')
def test_pending_not_expired(self, now):
    now.return_value = 0
    request = self.request()
    now.return_value = self.timeout - 1
    self.assertFalse(request.expired)