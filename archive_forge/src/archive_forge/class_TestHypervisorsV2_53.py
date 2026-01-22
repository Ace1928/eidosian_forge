from novaclient.tests.functional.v2.legacy import test_hypervisors
class TestHypervisorsV2_53(TestHypervisorsV28):
    COMPUTE_API_VERSION = '2.53'

    def test_list(self):
        self._test_list(cpu_info_type=dict, uuid_as_id=True)

    def test_search_with_details(self):
        hypervisors = self.client.hypervisors.list()
        hypervisor = hypervisors[0]
        hypervisors = self.client.hypervisors.search(hypervisor.hypervisor_hostname, detailed=True)
        self.assertEqual(1, len(hypervisors))
        hypervisor = hypervisors[0]
        self.assertIsNotNone(hypervisor.service, 'Expected service in hypervisor: %s' % hypervisor)