from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_rebind_list_bad_value(self):
    self.assertRaisesRegex(TypeError, '^Invalid rebind value', MyTask, rebind=object())