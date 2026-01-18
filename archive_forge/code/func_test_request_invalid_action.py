from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_request_invalid_action(self):
    request = pr.Request(utils.DummyTask('hi'), uuidutils.generate_uuid(), pr.EXECUTE, {}, 1.0)
    request = request.to_dict()
    request['action'] = 'NOTHING'
    self.assertRaises(excp.InvalidFormat, pr.Request.validate, request)