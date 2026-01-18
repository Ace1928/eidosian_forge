from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_project_id_and_project_name(self, client_mock):
    self.shell('--os-project-name=fake_tenant --os-project-id=123tenant workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('fake_tenant', params[1]['project_name'])
    self.assertEqual('123tenant', params[1]['project_id'])