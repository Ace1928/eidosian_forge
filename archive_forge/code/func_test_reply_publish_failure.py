from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
@mock.patch('taskflow.engines.worker_based.server.LOG.critical')
def test_reply_publish_failure(self, mocked_exception):
    self.proxy_inst_mock.publish.side_effect = RuntimeError('Woot!')
    s = self.server(reset_master_mock=True)
    s._reply(True, self.reply_to, self.task_uuid)
    self.master_mock.assert_has_calls([mock.call.Response(pr.FAILURE), mock.call.proxy.publish(self.response_inst_mock, self.reply_to, correlation_id=self.task_uuid)])
    self.assertTrue(mocked_exception.called)