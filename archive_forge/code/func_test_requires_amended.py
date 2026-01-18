from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_requires_amended(self):
    my_task = MyTask(requires=('spam', 'eggs'))
    expected = {'spam': 'spam', 'eggs': 'eggs', 'context': 'context'}
    self.assertEqual(expected, my_task.rebind)