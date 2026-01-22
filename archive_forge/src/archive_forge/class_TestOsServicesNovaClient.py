from novaclient.tests.functional import base
class TestOsServicesNovaClient(base.ClientTestBase):
    """Functional tests for os-services attributes"""
    COMPUTE_API_VERSION = '2.1'

    def test_os_services_list(self):
        table = self.nova('service-list')
        for serv in self.client.services.list():
            self.assertIn(serv.binary, table)

    def test_os_service_disable_enable(self):
        for serv in self.client.services.list():
            if serv.binary != 'nova-compute':
                continue
            host = self._get_column_value_from_single_row_table(self.nova('service-list --binary %s' % serv.binary), 'Host')
            service = self.nova('service-disable %s' % host)
            self.addCleanup(self.nova, 'service-enable', params=host)
            status = self._get_column_value_from_single_row_table(service, 'Status')
            self.assertEqual('disabled', status)
            service = self.nova('service-enable %s' % host)
            status = self._get_column_value_from_single_row_table(service, 'Status')
            self.assertEqual('enabled', status)

    def test_os_service_disable_log_reason(self):
        for serv in self.client.services.list():
            if serv.binary != 'nova-compute':
                continue
            host = self._get_column_value_from_single_row_table(self.nova('service-list --binary %s' % serv.binary), 'Host')
            service = self.nova('service-disable --reason test_disable %s' % host)
            self.addCleanup(self.nova, 'service-enable', params=host)
            status = self._get_column_value_from_single_row_table(service, 'Status')
            log_reason = self._get_column_value_from_single_row_table(service, 'Disabled Reason')
            self.assertEqual('disabled', status)
            self.assertEqual('test_disable', log_reason)