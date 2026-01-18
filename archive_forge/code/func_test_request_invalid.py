from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_request_invalid(self):
    msg = {'task_name': 1, 'task_cls': False, 'arguments': []}
    self.assertRaises(excp.InvalidFormat, pr.Request.validate, msg)