from manilaclient.tests.functional.osc import base
def test_openstack_share_network_create(self):
    share_network_name = 'test_create_share_network'
    share_network = self.create_share_network(name=share_network_name)
    self.assertEqual(share_network['name'], share_network_name)
    share_network_list = self.listing_result('share network', 'list')
    self.assertIn(share_network['id'], [item['ID'] for item in share_network_list])