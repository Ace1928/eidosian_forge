from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_service_type(self, client_mock):
    self.shell('--os-mistral-service-type=test workbook-list')
    self.assertTrue(client_mock.called)
    parmters = client_mock.call_args
    self.assertEqual('test', parmters[1]['service_type'])