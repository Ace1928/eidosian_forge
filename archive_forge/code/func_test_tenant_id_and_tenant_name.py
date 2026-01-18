from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_tenant_id_and_tenant_name(self, client_mock):
    self.shell('--os-tenant-id=123tenant --os-tenant-name=fake_tenant workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('fake_tenant', params[1]['project_name'])
    self.assertEqual('123tenant', params[1]['project_id'])