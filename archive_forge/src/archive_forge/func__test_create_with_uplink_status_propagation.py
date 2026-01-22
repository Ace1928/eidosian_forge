from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def _test_create_with_uplink_status_propagation(self, enable=True):
    arglist = ['--network', self._port.network_id, 'test-port']
    if enable:
        arglist += ['--enable-uplink-status-propagation']
    else:
        arglist += ['--disable-uplink-status-propagation']
    verifylist = [('network', self._port.network_id), ('name', 'test-port')]
    if enable:
        verifylist.append(('enable_uplink_status_propagation', True))
    else:
        verifylist.append(('disable_uplink_status_propagation', True))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'propagate_uplink_status': enable, 'name': 'test-port'})
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)