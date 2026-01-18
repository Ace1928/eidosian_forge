from taskflow.engines.worker_based import endpoint as ep
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import server
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
def test_parse_request_with_failures(self):
    failures = {'0': failure.Failure.from_exception(Exception('test1')), '1': failure.Failure.from_exception(Exception('test2'))}
    request = self.make_request(action='revert', failures=failures)
    bundle = pr.Request.from_dict(request)
    task_cls, task_name, action, task_args = bundle
    self.assertEqual((self.task.name, self.task.name, 'revert', dict(arguments=self.task_args, failures=dict(((i, utils.FailureMatcher(f)) for i, f in failures.items())))), (task_cls, task_name, action, task_args))