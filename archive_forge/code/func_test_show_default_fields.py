from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_show_default_fields(self):
    rows = ['console_enabled', 'clean_step', 'created_at', 'deploy_step', 'driver', 'driver_info', 'driver_internal_info', 'extra', 'inspection_finished_at', 'inspection_started_at', 'instance_info', 'instance_uuid', 'last_error', 'maintenance', 'maintenance_reason', 'name', 'power_state', 'properties', 'provision_state', 'provision_updated_at', 'reservation', 'target_power_state', 'target_provision_state', 'updated_at', 'uuid']
    node_show = self.openstack('baremetal node show {}'.format(self.node['uuid']))
    nodes_show_rows = self._get_table_rows(node_show)
    self.assertTrue(set(rows).issubset(set(nodes_show_rows)))