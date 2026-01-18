from unittest import mock
from openstack.block_storage.v2 import _proxy
from openstack.block_storage.v2 import backup
from openstack.block_storage.v2 import capabilities
from openstack.block_storage.v2 import limits
from openstack.block_storage.v2 import quota_set
from openstack.block_storage.v2 import snapshot
from openstack.block_storage.v2 import stats
from openstack.block_storage.v2 import type
from openstack.block_storage.v2 import volume
from openstack import resource
from openstack.tests.unit import test_proxy_base
def test_volume_wait_for(self):
    value = volume.Volume(id='1234')
    self.verify_wait_for_status(self.proxy.wait_for_status, method_args=[value], expected_args=[self.proxy, value, 'available', ['error'], 2, 120], expected_kwargs={'callback': None})