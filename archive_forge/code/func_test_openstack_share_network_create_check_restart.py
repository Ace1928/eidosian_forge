from manilaclient.tests.functional.osc import base
def test_openstack_share_network_create_check_restart(self):
    share_network = self.create_share_network()
    check_result = self.check_create_network_subnet(share_network['id'], restart_check=True)
    self.assertEqual('True', check_result['compatible'])
    self.assertEqual('{}', check_result['hosts_check_result'])