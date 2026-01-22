import uuid
from openstackclient.tests.functional.network.v2 import common
def _test_set_router_distributed(self, router_name):
    if not self.is_extension_enabled('dvr'):
        return
    cmd_output = self.openstack('router set ' + '--distributed ' + '--external-gateway public ' + router_name)
    self.assertOutput('', cmd_output)
    cmd_output = self.openstack('router show ' + router_name, parse_output=True)
    self.assertTrue(cmd_output['distributed'])
    self.assertIsNotNone(cmd_output['external_gateway_info'])