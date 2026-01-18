from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_revert_not_callable(self):
    self.assertRaises(ValueError, task.FunctorTask, lambda: None, revert=2)