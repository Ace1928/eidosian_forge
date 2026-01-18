from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
def test_queue_get(self):
    self.verify_get(self.proxy.get_queue, queue.Queue)
    self.verify_get_overrided(self.proxy, queue.Queue, 'openstack.message.v2.queue.Queue')