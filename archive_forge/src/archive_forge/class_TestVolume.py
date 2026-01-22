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
class TestVolume(TestVolumeProxy):

    def test_volume_get(self):
        self.verify_get(self.proxy.get_volume, volume.Volume)

    def test_volume_find(self):
        self.verify_find(self.proxy.find_volume, volume.Volume, method_kwargs={'all_projects': True}, expected_kwargs={'list_base_path': '/volumes/detail', 'all_projects': True})

    def test_volumes_detailed(self):
        self.verify_list(self.proxy.volumes, volume.Volume, method_kwargs={'details': True, 'all_projects': True}, expected_kwargs={'base_path': '/volumes/detail', 'all_projects': True})

    def test_volumes_not_detailed(self):
        self.verify_list(self.proxy.volumes, volume.Volume, method_kwargs={'details': False, 'all_projects': True}, expected_kwargs={'all_projects': True})

    def test_volume_create_attrs(self):
        self.verify_create(self.proxy.create_volume, volume.Volume)

    def test_volume_delete(self):
        self.verify_delete(self.proxy.delete_volume, volume.Volume, False)

    def test_volume_delete_ignore(self):
        self.verify_delete(self.proxy.delete_volume, volume.Volume, True)

    def test_volume_delete_force(self):
        self._verify('openstack.block_storage.v2.volume.Volume.force_delete', self.proxy.delete_volume, method_args=['value'], method_kwargs={'force': True}, expected_args=[self.proxy])

    def test_get_volume_metadata(self):
        self._verify('openstack.block_storage.v2.volume.Volume.fetch_metadata', self.proxy.get_volume_metadata, method_args=['value'], expected_args=[self.proxy], expected_result=volume.Volume(id='value', metadata={}))

    def test_set_volume_metadata(self):
        kwargs = {'a': '1', 'b': '2'}
        id = 'an_id'
        self._verify('openstack.block_storage.v2.volume.Volume.set_metadata', self.proxy.set_volume_metadata, method_args=[id], method_kwargs=kwargs, method_result=volume.Volume.existing(id=id, metadata=kwargs), expected_args=[self.proxy], expected_kwargs={'metadata': kwargs}, expected_result=volume.Volume.existing(id=id, metadata=kwargs))

    def test_delete_volume_metadata(self):
        self._verify('openstack.block_storage.v2.volume.Volume.delete_metadata_item', self.proxy.delete_volume_metadata, expected_result=None, method_args=['value', ['key']], expected_args=[self.proxy, 'key'])

    def test_backend_pools(self):
        self.verify_list(self.proxy.backend_pools, stats.Pools)

    def test_volume_wait_for(self):
        value = volume.Volume(id='1234')
        self.verify_wait_for_status(self.proxy.wait_for_status, method_args=[value], expected_args=[self.proxy, value, 'available', ['error'], 2, 120], expected_kwargs={'callback': None})