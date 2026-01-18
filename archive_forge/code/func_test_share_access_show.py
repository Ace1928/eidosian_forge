import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
def test_share_access_show(self):
    share = self.create_share()
    access_rule = self.create_share_access_rule(share=share['name'], access_type='ip', access_to='0.0.0.0/0', wait=True)
    access_rule_show = self.dict_result('share', f'access show {access_rule['id']}')
    self.assertEqual(access_rule_show['id'], access_rule['id'])
    self.assertEqual(access_rule_show['share_id'], share['id'])
    self.assertEqual(access_rule_show['access_level'], 'rw')
    self.assertEqual(access_rule_show['access_to'], '0.0.0.0/0')
    self.assertEqual(access_rule_show['access_type'], 'ip')
    self.assertEqual(access_rule_show['state'], 'active')
    self.assertEqual(access_rule_show['access_key'], 'None')
    self.assertEqual(access_rule_show['created_at'], access_rule['created_at'])
    self.assertEqual(access_rule_show['properties'], '')
    self.assertIn('updated_at', access_rule_show)