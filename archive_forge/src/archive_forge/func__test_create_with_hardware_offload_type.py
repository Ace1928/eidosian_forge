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
def _test_create_with_hardware_offload_type(self, hwol_type=None):
    arglist = ['--network', self._port.network_id, 'test-port']
    if hwol_type:
        arglist += ['--hardware-offload-type', hwol_type]
    hardware_offload_type = None if not hwol_type else hwol_type
    verifylist = [('network', self._port.network_id), ('name', 'test-port')]
    if hwol_type:
        verifylist.append(('hardware_offload_type', hwol_type))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    create_args = {'admin_state_up': True, 'network_id': self._port.network_id, 'name': 'test-port'}
    if hwol_type:
        create_args['hardware_offload_type'] = hardware_offload_type
    self.network_client.create_port.assert_called_once_with(**create_args)
    self.assertEqual(set(self.columns), set(columns))
    self.assertCountEqual(self.data, data)