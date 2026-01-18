import uuid
from testtools import matchers
from openstack.tests.unit import base
def test_update_endpoint_v3(self):
    service_data = self._get_service_data()
    dummy_url = self._dummy_url()
    endpoint_data = self._get_endpoint_v3_data(service_id=service_data.service_id, interface='admin', enabled=False)
    reference_request = endpoint_data.json_request.copy()
    reference_request['endpoint']['url'] = dummy_url
    self.register_uris([dict(method='PATCH', uri=self.get_mock_url(append=[endpoint_data.endpoint_id]), status_code=200, json=endpoint_data.json_response, validate=dict(json=reference_request))])
    endpoint = self.cloud.update_endpoint(endpoint_data.endpoint_id, service_name_or_id=service_data.service_id, region=endpoint_data.region_id, url=dummy_url, interface=endpoint_data.interface, enabled=False)
    self.assertThat(endpoint.id, matchers.Equals(endpoint_data.endpoint_id))
    self.assertThat(endpoint.service_id, matchers.Equals(service_data.service_id))
    self.assertThat(endpoint.url, matchers.Equals(endpoint_data.url))
    self.assertThat(endpoint.interface, matchers.Equals(endpoint_data.interface))
    self.assert_calls()