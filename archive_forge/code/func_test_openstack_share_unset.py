from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_unset(self):
    share = self.create_share(name='test_name', properties={'foo': 'bar', 'test_key': 'test_value'})
    result1 = self.dict_result('share', f'show {share['id']}')
    self.assertEqual(share['id'], result1['id'])
    self.assertEqual(share['name'], result1['name'])
    self.assertEqual("foo='bar', test_key='test_value'", result1['properties'])
    self.openstack(f'share unset {share['id']} --name --property test_key')
    result2 = self.dict_result('share', f'show {share['id']}')
    self.assertEqual(share['id'], result2['id'])
    self.assertEqual('None', result2['name'])
    self.assertEqual("foo='bar'", result2['properties'])