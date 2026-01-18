import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
@mock.patch('oslo_utils.timeutils.now')
def test_on_wait_task_expired(self, mock_now):
    mock_now.side_effect = [0, 120]
    self.request_inst_mock.expired = True
    self.request_inst_mock.created_on = 0
    ex = self.executor()
    ex._ongoing_requests[self.task_uuid] = self.request_inst_mock
    self.assertEqual(1, len(ex._ongoing_requests))
    ex._on_wait()
    self.assertEqual(0, len(ex._ongoing_requests))