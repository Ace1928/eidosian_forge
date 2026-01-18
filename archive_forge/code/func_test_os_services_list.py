from novaclient.tests.functional import base
def test_os_services_list(self):
    table = self.nova('service-list')
    for serv in self.client.services.list():
        self.assertIn(serv.binary, table)