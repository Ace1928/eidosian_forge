from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_search_services(self):
    service_data = self._get_service_data()
    service2_data = self._get_service_data(type=service_data.service_type)
    self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]})])
    services = self.cloud.search_services(name_or_id=service_data.service_id)
    self.assertThat(len(services), matchers.Equals(1))
    self.assertThat(services[0].id, matchers.Equals(service_data.service_id))
    services = self.cloud.search_services(name_or_id=service_data.service_name)
    self.assertThat(len(services), matchers.Equals(1))
    self.assertThat(services[0].name, matchers.Equals(service_data.service_name))
    services = self.cloud.search_services(name_or_id='!INVALID!')
    self.assertThat(len(services), matchers.Equals(0))
    services = self.cloud.search_services(filters={'type': service_data.service_type})
    self.assertThat(len(services), matchers.Equals(2))
    self.assertThat(services[0].id, matchers.Equals(service_data.service_id))
    self.assertThat(services[1].id, matchers.Equals(service2_data.service_id))
    self.assert_calls()