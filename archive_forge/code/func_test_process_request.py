from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_process_request(self):
    s = self.server(reset_master_mock=True)
    s._process_request(self.make_request(), self.message_mock)
    master_mock_calls = [mock.call.Response(pr.RUNNING), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid), mock.call.Response(pr.SUCCESS, result=1), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid)]
    self.master_mock.assert_has_calls(master_mock_calls)