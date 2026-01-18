from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_task_repr(self):
    my_task = MyTask(name='my')
    self.assertEqual('<%s.MyTask "my==1.0">' % __name__, repr(my_task))