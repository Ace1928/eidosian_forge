from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class SeparateRevertTask(task.Task):

    def execute(self, execute_arg):
        pass

    def revert(self, revert_arg, result, flow_failures):
        pass