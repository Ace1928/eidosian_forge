from unittest import mock
from openstack.shared_file_system.v2 import _proxy
from openstack.shared_file_system.v2 import limit
from openstack.shared_file_system.v2 import resource_locks
from openstack.shared_file_system.v2 import share
from openstack.shared_file_system.v2 import share_access_rule
from openstack.shared_file_system.v2 import share_group
from openstack.shared_file_system.v2 import share_group_snapshot
from openstack.shared_file_system.v2 import share_instance
from openstack.shared_file_system.v2 import share_network
from openstack.shared_file_system.v2 import share_network_subnet
from openstack.shared_file_system.v2 import share_snapshot
from openstack.shared_file_system.v2 import share_snapshot_instance
from openstack.shared_file_system.v2 import storage_pool
from openstack.shared_file_system.v2 import user_message
from openstack.tests.unit import test_proxy_base
class TestShareSnapshotInstanceResource(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestShareSnapshotInstanceResource, self).setUp()
        self.proxy = _proxy.Proxy(self.session)

    def test_share_snapshot_instances(self):
        self.verify_list(self.proxy.share_snapshot_instances, share_snapshot_instance.ShareSnapshotInstance)

    def test_share_snapshot_instance_detailed(self):
        self.verify_list(self.proxy.share_snapshot_instances, share_snapshot_instance.ShareSnapshotInstance, method_kwargs={'details': True, 'query': {'snapshot_id': 'fake'}}, expected_kwargs={'query': {'snapshot_id': 'fake'}})

    def test_share_snapshot_instance_not_detailed(self):
        self.verify_list(self.proxy.share_snapshot_instances, share_snapshot_instance.ShareSnapshotInstance, method_kwargs={'details': False, 'query': {'snapshot_id': 'fake'}}, expected_kwargs={'query': {'snapshot_id': 'fake'}})

    def test_share_snapshot_instance_get(self):
        self.verify_get(self.proxy.get_share_snapshot_instance, share_snapshot_instance.ShareSnapshotInstance)