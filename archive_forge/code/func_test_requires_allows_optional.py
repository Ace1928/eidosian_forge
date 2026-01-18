from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_requires_allows_optional(self):
    my_task = DefaultArgTask(requires=('spam', 'eggs'))
    self.assertEqual(set(['spam', 'eggs']), my_task.requires)
    self.assertEqual(set(), my_task.optional)