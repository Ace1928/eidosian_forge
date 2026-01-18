import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_on_message_response_state_progress(self):
    response = pr.Response(pr.EVENT, event_type=task_atom.EVENT_UPDATE_PROGRESS, details={'progress': 1.0})
    ex = self.executor()
    ex._ongoing_requests[self.task_uuid] = self.request_inst_mock
    ex._process_response(response.to_dict(), self.message_mock)
    expected_calls = [mock.call.task.notifier.notify(task_atom.EVENT_UPDATE_PROGRESS, {'progress': 1.0})]
    self.assertEqual(expected_calls, self.request_inst_mock.mock_calls)