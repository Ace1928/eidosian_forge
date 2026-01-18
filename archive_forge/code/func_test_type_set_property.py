from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_type
def test_type_set_property(self):
    arglist = ['--property', 'myprop=myvalue', '--multiattach', '--cacheable', '--replicated', '--availability-zone', 'az1', self.volume_type.id]
    verifylist = [('name', None), ('description', None), ('properties', {'myprop': 'myvalue'}), ('multiattach', True), ('cacheable', True), ('replicated', True), ('availability_zones', ['az1']), ('volume_type', self.volume_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)
    self.volume_type.set_keys.assert_called_once_with({'myprop': 'myvalue', 'multiattach': '<is> True', 'cacheable': '<is> True', 'replication_enabled': '<is> True', 'RESKEY:availability_zones': 'az1'})
    self.volume_type_access_mock.add_project_access.assert_not_called()
    self.volume_encryption_types_mock.update.assert_not_called()
    self.volume_encryption_types_mock.create.assert_not_called()