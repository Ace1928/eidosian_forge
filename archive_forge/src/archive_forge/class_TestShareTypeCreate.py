from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_types as osc_share_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTypeCreate(TestShareType):

    def setUp(self):
        super(TestShareTypeCreate, self).setUp()
        self.new_share_type = manila_fakes.FakeShareType.create_one_sharetype()
        self.shares_mock.create.return_value = self.new_share_type
        self.cmd = osc_share_types.CreateShareType(self.app, None)
        self.data = [self.new_share_type.id, self.new_share_type.name, 'public', self.new_share_type.is_default, 'driver_handles_share_servers : True', 'replication_type : readable\nmount_snapshot_support : False\nrevert_to_snapshot_support : False\ncreate_share_from_snapshot_support : True\nsnapshot_support : True', self.new_share_type.description]
        self.raw_data = [self.new_share_type.id, self.new_share_type.name, 'public', self.new_share_type.is_default, {'driver_handles_share_servers': True}, {'replication_type': 'readable', 'mount_snapshot_support': False, 'revert_to_snapshot_support': False, 'create_share_from_snapshot_support': True, 'snapshot_support': True}, self.new_share_type.description]

    def test_share_type_create_required_args(self):
        """Verifies required arguments."""
        arglist = [self.new_share_type.name, 'True']
        verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'True')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(extra_specs={}, is_public=True, name=self.new_share_type.name, spec_driver_handles_share_servers=True)
        self.assertCountEqual(COLUMNS, columns)
        self.assertCountEqual(self.data, data)

    def test_share_type_create_json_fomrat(self):
        """Verifies --format json."""
        arglist = [self.new_share_type.name, 'True', '-f', 'json']
        verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'True')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(extra_specs={}, is_public=True, name=self.new_share_type.name, spec_driver_handles_share_servers=True)
        self.assertCountEqual(COLUMNS, columns)
        self.assertCountEqual(self.raw_data, data)

    def test_share_type_create_missing_required_arg(self):
        """Verifies missing required arguments."""
        arglist = [self.new_share_type.name]
        verifylist = [('name', self.new_share_type.name)]
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_type_create_private(self):
        arglist = [self.new_share_type.name, 'True', '--public', 'False']
        verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'True'), ('public', 'False')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(extra_specs={}, is_public=False, name=self.new_share_type.name, spec_driver_handles_share_servers=True)
        self.assertCountEqual(COLUMNS, columns)
        self.assertCountEqual(self.data, data)

    def test_share_type_create_extra_specs(self):
        arglist = [self.new_share_type.name, 'True', '--extra-specs', 'snapshot_support=true']
        verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'True'), ('extra_specs', ['snapshot_support=true'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(extra_specs={'snapshot_support': 'True'}, is_public=True, name=self.new_share_type.name, spec_driver_handles_share_servers=True)
        self.assertCountEqual(COLUMNS, columns)
        self.assertCountEqual(self.data, data)

    def test_share_type_create_dhss_invalid_value(self):
        arglist = [self.new_share_type.name, 'non_bool_value']
        verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'non_bool_value')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_type_create_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.40')
        arglist = [self.new_share_type.name, 'True', '--description', 'Description']
        verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'True'), ('description', 'Description')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_type_create_dhss_defined_twice(self):
        arglist = [self.new_share_type.name, 'True', '--extra-specs', 'driver_handles_share_servers=true']
        verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'True'), ('extra_specs', ['driver_handles_share_servers=true'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_type_create_bool_args(self):
        arglist = [self.new_share_type.name, 'True', '--snapshot-support', 'true']
        verifylist = [('name', self.new_share_type.name), ('spec_driver_handles_share_servers', 'True'), ('snapshot_support', 'true')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(extra_specs={'snapshot_support': 'True'}, is_public=True, name=self.new_share_type.name, spec_driver_handles_share_servers=True)
        self.assertCountEqual(COLUMNS, columns)
        self.assertCountEqual(self.data, data)