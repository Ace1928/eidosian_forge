from unittest import mock
from openstack.block_storage.v3 import _proxy
from openstack.block_storage.v3 import backup
from openstack.block_storage.v3 import capabilities
from openstack.block_storage.v3 import extension
from openstack.block_storage.v3 import group
from openstack.block_storage.v3 import group_snapshot
from openstack.block_storage.v3 import group_type
from openstack.block_storage.v3 import limits
from openstack.block_storage.v3 import quota_set
from openstack.block_storage.v3 import resource_filter
from openstack.block_storage.v3 import service
from openstack.block_storage.v3 import snapshot
from openstack.block_storage.v3 import stats
from openstack.block_storage.v3 import type
from openstack.block_storage.v3 import volume
from openstack import resource
from openstack.tests.unit import test_proxy_base
def test_volume_revert_to_snapshot(self):
    self._verify('openstack.block_storage.v3.volume.Volume.revert_to_snapshot', self.proxy.revert_volume_to_snapshot, method_args=['value', '1'], expected_args=[self.proxy, '1'])