from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_transfer_request
def test_transfer_create_with_name(self):
    arglist = ['--name', self.volume_transfer.name, self.volume.id]
    verifylist = [('name', self.volume_transfer.name), ('volume', self.volume.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.transfer_mock.create.assert_called_once_with(self.volume.id, self.volume_transfer.name)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)