from unittest import mock
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_volume
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
def test_server_volume_update(self):
    arglist = [self.server.id, self.volume.id]
    verifylist = [('server', self.server.id), ('volume', self.volume.id), ('delete_on_termination', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.update_volume_attachment.assert_not_called()
    self.assertIsNone(result)