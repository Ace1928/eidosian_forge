from novaclient.tests.functional.v2.legacy import test_hypervisors
def test_search_with_details(self):
    hypervisors = self.client.hypervisors.list()
    hypervisor = hypervisors[0]
    hypervisors = self.client.hypervisors.search(hypervisor.hypervisor_hostname, detailed=True)
    self.assertEqual(1, len(hypervisors))
    hypervisor = hypervisors[0]
    self.assertIsNotNone(hypervisor.service, 'Expected service in hypervisor: %s' % hypervisor)