from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_task_str(self):
    my_task = MyTask(name='my')
    self.assertEqual('"my==1.0"', str(my_task))