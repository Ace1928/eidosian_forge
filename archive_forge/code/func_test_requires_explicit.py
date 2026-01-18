from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_requires_explicit(self):
    my_task = MyTask(auto_extract=False, requires=('spam', 'eggs', 'context'))
    expected = {'spam': 'spam', 'eggs': 'eggs', 'context': 'context'}
    self.assertEqual(expected, my_task.rebind)