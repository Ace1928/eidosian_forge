from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_request_transitions(self):
    request = self.request()
    self.assertEqual(pr.WAITING, request.current_state)
    self.assertIn(request.current_state, pr.WAITING_STATES)
    self.assertRaises(excp.InvalidState, request.transition, pr.SUCCESS)
    self.assertFalse(request.transition(pr.WAITING))
    self.assertTrue(request.transition(pr.PENDING))
    self.assertTrue(request.transition(pr.RUNNING))
    self.assertTrue(request.transition(pr.SUCCESS))
    for s in (pr.PENDING, pr.WAITING):
        self.assertRaises(excp.InvalidState, request.transition, s)