from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
def test_subscriptions(self):
    self.verify_list(self.proxy.subscriptions, subscription.Subscription, method_kwargs={'queue_name': 'test_queue'}, expected_kwargs={'queue_name': 'test_queue'})