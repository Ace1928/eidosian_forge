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
class TestShareGroupResource(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestShareGroupResource, self).setUp()
        self.proxy = _proxy.Proxy(self.session)

    def test_share_groups(self):
        self.verify_list(self.proxy.share_groups, share_group.ShareGroup)

    def test_share_groups_query(self):
        self.verify_list(self.proxy.share_groups, share_group.ShareGroup, method_kwargs={'query': 1}, expected_kwargs={'query': 1})

    def test_share_group_get(self):
        self.verify_get(self.proxy.get_share_group, share_group.ShareGroup)

    def test_share_group_find(self):
        self.verify_find(self.proxy.find_share_group, share_group.ShareGroup)

    def test_share_group_delete(self):
        self.verify_delete(self.proxy.delete_share_group, share_group.ShareGroup, False)

    def test_share_group_delete_ignore(self):
        self.verify_delete(self.proxy.delete_share_group, share_group.ShareGroup, True)

    def test_share_group_create(self):
        self.verify_create(self.proxy.create_share_group, share_group.ShareGroup)

    def test_share_group_update(self):
        self.verify_update(self.proxy.update_share_group, share_group.ShareGroup)

    def test_share_group_snapshots(self):
        self.verify_list(self.proxy.share_group_snapshots, share_group_snapshot.ShareGroupSnapshot)

    def test_share_group_snapshot_get(self):
        self.verify_get(self.proxy.get_share_group_snapshot, share_group_snapshot.ShareGroupSnapshot)

    def test_share_group_snapshot_update(self):
        self.verify_update(self.proxy.update_share_group_snapshot, share_group_snapshot.ShareGroupSnapshot)

    def test_share_group_snapshot_delete(self):
        self.verify_delete(self.proxy.delete_share_group_snapshot, share_group_snapshot.ShareGroupSnapshot, False)

    def test_share_group_snapshot_delete_ignore(self):
        self.verify_delete(self.proxy.delete_share_group_snapshot, share_group_snapshot.ShareGroupSnapshot, True)