from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_register_handler_is_none(self):
    a_task = MyTask()
    self.assertRaises(ValueError, a_task.notifier.register, task.EVENT_UPDATE_PROGRESS, None)
    self.assertEqual(0, len(a_task.notifier))