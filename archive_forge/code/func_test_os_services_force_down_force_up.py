from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_os_services
from novaclient import utils
def test_os_services_force_down_force_up(self):
    for serv in self.client.services.list():
        if serv.binary != 'nova-compute':
            continue
        service = self.nova('service-force-down %s' % serv.id)
        self.addCleanup(self.nova, 'service-force-down --unset', params='%s' % serv.id)
        service_id = self._get_column_value_from_single_row_table(service, 'ID')
        self.assertEqual(serv.id, service_id)
        forced_down = self._get_column_value_from_single_row_table(service, 'Forced down')
        self.assertEqual('True', forced_down)
        service = self.nova('service-force-down --unset %s' % serv.id)
        forced_down = self._get_column_value_from_single_row_table(service, 'Forced down')
        self.assertEqual('False', forced_down)