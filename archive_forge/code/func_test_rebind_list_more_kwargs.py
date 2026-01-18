from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_rebind_list_more_kwargs(self):
    my_task = KwargsTask(rebind=('a', 'b', 'c'))
    expected = {'spam': 'a', 'b': 'b', 'c': 'c'}
    self.assertEqual(expected, my_task.rebind)
    self.assertEqual(set(['a', 'b', 'c']), my_task.requires)