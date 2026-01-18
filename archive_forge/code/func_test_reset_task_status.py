import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
def test_reset_task_status(self):
    self._mock_action()
    self.management.reset_task_status(1)
    self.assertEqual(1, self.management._action.call_count)
    self.assertEqual({'reset-task-status': {}}, self.body_)