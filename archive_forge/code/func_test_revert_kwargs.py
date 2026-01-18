from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_revert_kwargs(self):
    my_task = RevertKwargsTask()
    expected_rebind = {'execute_arg1': 'execute_arg1', 'execute_arg2': 'execute_arg2'}
    self.assertEqual(expected_rebind, my_task.rebind)
    expected_rebind = {'execute_arg1': 'execute_arg1'}
    self.assertEqual(expected_rebind, my_task.revert_rebind)
    self.assertEqual(set(['execute_arg1', 'execute_arg2']), my_task.requires)