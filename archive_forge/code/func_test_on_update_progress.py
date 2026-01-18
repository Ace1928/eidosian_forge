from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_on_update_progress(self):
    request = self.make_request(task=utils.ProgressingTask(), arguments={})
    s = self.server(reset_master_mock=True)
    s._process_request(request, self.message_mock)
    master_mock_calls = [mock.call.Response(pr.RUNNING), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid), mock.call.Response(pr.EVENT, details={'progress': 0.0}, event_type=task_atom.EVENT_UPDATE_PROGRESS), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid), mock.call.Response(pr.EVENT, details={'progress': 1.0}, event_type=task_atom.EVENT_UPDATE_PROGRESS), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid), mock.call.Response(pr.SUCCESS, result=5), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid)]
    self.master_mock.assert_has_calls(master_mock_calls)