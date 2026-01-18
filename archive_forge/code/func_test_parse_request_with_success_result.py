from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_parse_request_with_success_result(self):
    request = self.make_request(action='revert', result=1)
    bundle = pr.Request.from_dict(request)
    task_cls, task_name, action, task_args = bundle
    self.assertEqual((self.task.name, self.task.name, 'revert', dict(arguments=self.task_args, result=1)), (task_cls, task_name, action, task_args))