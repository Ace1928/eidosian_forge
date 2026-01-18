from openstackclient.tests.functional.identity.v2 import common
def test_service_multi_delete(self):
    service_name_1 = self._create_dummy_service(add_clean_up=False)
    service_name_2 = self._create_dummy_service(add_clean_up=False)
    raw_output = self.openstack('service delete ' + service_name_1 + ' ' + service_name_2)
    self.assertEqual(0, len(raw_output))