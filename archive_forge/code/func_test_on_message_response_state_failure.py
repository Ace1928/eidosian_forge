import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_on_message_response_state_failure(self):
    a_failure = failure.Failure.from_exception(Exception('test'))
    failure_dict = a_failure.to_dict()
    response = pr.Response(pr.FAILURE, result=failure_dict)
    ex = self.executor()
    ex._ongoing_requests[self.task_uuid] = self.request_inst_mock
    ex._process_response(response.to_dict(), self.message_mock)
    self.assertEqual(0, len(ex._ongoing_requests))
    expected_calls = [mock.call.transition_and_log_error(pr.FAILURE, logger=mock.ANY), mock.call.set_result(result=test_utils.FailureMatcher(a_failure))]
    self.assertEqual(expected_calls, self.request_inst_mock.mock_calls)