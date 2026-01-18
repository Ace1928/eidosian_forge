from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_rebind_list_all(self):
    my_task = MyTask(rebind=('a', 'b', 'c'))
    expected = {'context': 'a', 'spam': 'b', 'eggs': 'c'}
    self.assertEqual(expected, my_task.rebind)
    self.assertEqual(set(['a', 'b', 'c']), my_task.requires)