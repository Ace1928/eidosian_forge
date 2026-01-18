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
def test_type_list_with_default_option(self):
    arglist = ['--default']
    verifylist = [('encryption_type', False), ('long', False), ('is_public', None), ('default', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_types_mock.default.assert_called_once_with()
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data_with_default_type, list(data))