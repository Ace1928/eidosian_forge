from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_rebind_unknown(self):
    self.assertRaisesRegex(ValueError, '^Extra arguments', MyTask, rebind={'foo': 'bar'})