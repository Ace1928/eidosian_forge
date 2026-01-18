from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_parse_request(self):
    request = self.make_request()
    bundle = pr.Request.from_dict(request)
    task_cls, task_name, action, task_args = bundle
    self.assertEqual((self.task.name, self.task.name, self.task_action, dict(arguments=self.task_args)), (task_cls, task_name, action, task_args))