import copy
from openstack import exceptions
from openstack.tests.unit import base
def test_update_zone(self):
    fake_zone = ZoneTestWrapper(self, zone_dict)
    new_ttl = 7200
    updated_zone_dict = copy.copy(zone_dict)
    updated_zone_dict['ttl'] = new_ttl
    updated_zone = ZoneTestWrapper(self, updated_zone_dict)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id']]), json=fake_zone.get_get_response_json()), dict(method='PATCH', uri=self.get_mock_url('dns', 'public', append=['v2', 'zones', fake_zone['id']]), json=updated_zone.get_get_response_json(), validate=dict(json={'ttl': new_ttl}))])
    z = self.cloud.update_zone(fake_zone['id'], ttl=new_ttl)
    updated_zone.cmp(z)
    self.assert_calls()