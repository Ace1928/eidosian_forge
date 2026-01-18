import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
def test_set_share_access(self):
    share = self.create_share()
    access_rule = self.create_share_access_rule(share=share['name'], access_type='ip', access_to='0.0.0.0/0', wait=True)
    self.assertEqual(access_rule['properties'], '')
    self.openstack('share', params=f'access set {access_rule['id']} --property foo=bar')
    access_rule = self.dict_result('share', f'access show {access_rule['id']}')
    self.assertEqual(access_rule['properties'], 'foo : bar')