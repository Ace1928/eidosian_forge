from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_update_service_v3(self):
    service_data = self._get_service_data(name='a service', type='network', description='A test service')
    request = service_data.json_request.copy()
    request['enabled'] = False
    resp = service_data.json_response_v3.copy()
    resp['enabled'] = False
    request.pop('description')
    request.pop('name')
    request.pop('type')
    self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [resp['service']]}), dict(method='PATCH', uri=self.get_mock_url(append=[service_data.service_id]), status_code=200, json=resp, validate=dict(json={'service': request}))])
    service = self.cloud.update_service(service_data.service_id, enabled=False)
    self.assertThat(service.name, matchers.Equals(service_data.service_name))
    self.assertThat(service.id, matchers.Equals(service_data.service_id))
    self.assertThat(service.description, matchers.Equals(service_data.description))
    self.assertThat(service.type, matchers.Equals(service_data.service_type))
    self.assert_calls()