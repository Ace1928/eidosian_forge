from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_passed_name(self):
    my_task = MyTask(name='my name')
    self.assertEqual('my name', my_task.name)