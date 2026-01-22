from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
class TestMessageSubscription(TestMessageProxy):

    def test_subscription_create(self):
        self._verify('openstack.message.v2.subscription.Subscription.create', self.proxy.create_subscription, method_args=['test_queue'], expected_args=[self.proxy], expected_kwargs={'base_path': None})

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_subscription_get(self, mock_get_resource):
        mock_get_resource.return_value = 'resource_or_id'
        self._verify('openstack.proxy.Proxy._get', self.proxy.get_subscription, method_args=['test_queue', 'resource_or_id'], expected_args=[subscription.Subscription, 'resource_or_id'])
        mock_get_resource.assert_called_once_with(subscription.Subscription, 'resource_or_id', queue_name='test_queue')
        self.verify_get_overrided(self.proxy, subscription.Subscription, 'openstack.message.v2.subscription.Subscription')

    def test_subscriptions(self):
        self.verify_list(self.proxy.subscriptions, subscription.Subscription, method_kwargs={'queue_name': 'test_queue'}, expected_kwargs={'queue_name': 'test_queue'})

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_subscription_delete(self, mock_get_resource):
        mock_get_resource.return_value = 'test_subscription'
        self.verify_delete(self.proxy.delete_subscription, subscription.Subscription, ignore_missing=False, method_args=['test_queue', 'resource_or_id'], expected_args=['test_subscription'])
        mock_get_resource.assert_called_once_with(subscription.Subscription, 'resource_or_id', queue_name='test_queue')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_subscription_delete_ignore(self, mock_get_resource):
        mock_get_resource.return_value = 'test_subscription'
        self.verify_delete(self.proxy.delete_subscription, subscription.Subscription, ignore_missing=True, method_args=['test_queue', 'resource_or_id'], expected_args=['test_subscription'])
        mock_get_resource.assert_called_once_with(subscription.Subscription, 'resource_or_id', queue_name='test_queue')