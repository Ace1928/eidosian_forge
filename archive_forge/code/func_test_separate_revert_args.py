from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_separate_revert_args(self):
    my_task = SeparateRevertTask(rebind=('a',), revert_rebind=('b',))
    self.assertEqual({'execute_arg': 'a'}, my_task.rebind)
    self.assertEqual({'revert_arg': 'b'}, my_task.revert_rebind)
    self.assertEqual(set(['a', 'b']), my_task.requires)
    my_task = SeparateRevertTask(requires='execute_arg', revert_requires='revert_arg')
    self.assertEqual({'execute_arg': 'execute_arg'}, my_task.rebind)
    self.assertEqual({'revert_arg': 'revert_arg'}, my_task.revert_rebind)
    self.assertEqual(set(['execute_arg', 'revert_arg']), my_task.requires)