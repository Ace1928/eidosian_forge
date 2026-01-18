from openstackclient.tests.functional.volume.v2 import common
def test_volume_service_list(self):
    cmd_output = self.openstack('volume service list', parse_output=True)
    services = list(set([x['Binary'] for x in cmd_output]))
    hosts = list(set([x['Host'] for x in cmd_output]))
    cmd_output = self.openstack('volume service list ' + '--service ' + services[0], parse_output=True)
    for x in cmd_output:
        self.assertEqual(services[0], x['Binary'])
    cmd_output = self.openstack('volume service list ' + '--host ' + hosts[0], parse_output=True)
    for x in cmd_output:
        self.assertIn(hosts[0], x['Host'])