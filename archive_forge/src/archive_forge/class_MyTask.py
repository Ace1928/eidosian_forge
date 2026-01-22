from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
class MyTask(task.Task):

    def execute(self, context, spam, eggs):
        pass