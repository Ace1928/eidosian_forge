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
def test_set_port_that(self):
    arglist = ['--description', 'newDescription', '--enable', '--vnic-type', 'macvtap', '--binding-profile', 'foo=bar', '--host', 'binding-host-id-xxxx', '--name', 'newName', self._port.name]
    verifylist = [('description', 'newDescription'), ('enable', True), ('vnic_type', 'macvtap'), ('binding_profile', {'foo': 'bar'}), ('host', 'binding-host-id-xxxx'), ('name', 'newName'), ('port', self._port.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'admin_state_up': True, 'binding:vnic_type': 'macvtap', 'binding:profile': {'foo': 'bar'}, 'binding:host_id': 'binding-host-id-xxxx', 'description': 'newDescription', 'name': 'newName'}
    self.network_client.update_port.assert_called_once_with(self._port, **attrs)
    self.assertIsNone(result)