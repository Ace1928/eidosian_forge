from openstack.block_storage.v3 import backup
from openstack.tests.unit import base
def test_search_volume_backups(self):
    name = 'Volume1'
    vol1 = {'name': name, 'availability_zone': 'az1'}
    vol2 = {'name': name, 'availability_zone': 'az1'}
    vol3 = {'name': 'Volume2', 'availability_zone': 'az2'}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['backups', 'detail']), json={'backups': [vol1, vol2, vol3]})])
    result = self.cloud.search_volume_backups(name, {'availability_zone': 'az1'})
    self.assertEqual(len(result), 2)
    for a, b in zip([vol1, vol2], result):
        self._compare_backups(a, b)
    self.assert_calls()