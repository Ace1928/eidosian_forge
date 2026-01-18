from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_rebind_partial(self):
    my_task = MyTask(rebind={'spam': 'a', 'eggs': 'b'})
    expected = {'spam': 'a', 'eggs': 'b', 'context': 'context'}
    self.assertEqual(expected, my_task.rebind)
    self.assertEqual(set(['a', 'b', 'context']), my_task.requires)