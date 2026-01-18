import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
def test_share_access_allow(self):
    share = self.create_share()
    access_rule = self.create_share_access_rule(share=share['name'], access_type='ip', access_to='0.0.0.0/0', wait=True)
    self.assertEqual(access_rule['share_id'], share['id'])
    self.assertEqual(access_rule['state'], 'active')
    self.assertEqual(access_rule['access_type'], 'ip')
    self.assertEqual(access_rule['access_to'], '0.0.0.0/0')
    self.assertEqual(access_rule['properties'], '')
    self.assertEqual(access_rule['access_level'], 'rw')
    access_rules = self.listing_result('share', f'access list {share['id']}')
    self.assertIn(access_rule['id'], [item['ID'] for item in access_rules])
    access_rule = self.create_share_access_rule(share=share['name'], access_type='ip', access_to='12.34.56.78', access_level='ro', properties='foo=bar')
    self.assertEqual(access_rule['access_type'], 'ip')
    self.assertEqual(access_rule['access_to'], '12.34.56.78')
    self.assertEqual(access_rule['properties'], 'foo : bar')
    self.assertEqual(access_rule['access_level'], 'ro')