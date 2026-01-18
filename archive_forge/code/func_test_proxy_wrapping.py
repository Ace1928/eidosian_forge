from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def test_proxy_wrapping(self):
    proxy_factory = proxy.TaskFactory(self.factory, task_proxy_class=FakeProxy, task_proxy_kwargs={'dog': 'bark'})
    self.factory.new_task.return_value = 'fake_task'
    task = proxy_factory.new_task(type=self.fake_type, owner=self.fake_owner)
    self.factory.new_task.assert_called_once_with(type=self.fake_type, owner=self.fake_owner)
    self.assertIsInstance(task, FakeProxy)
    self.assertEqual('fake_task', task.base)