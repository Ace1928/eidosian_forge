from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareInstanceList(TestShareInstance):
    columns = ['id', 'share_id', 'host', 'status', 'availability_zone', 'share_network_id', 'share_server_id', 'share_type_id']
    column_headers = utils.format_column_headers(columns)

    def setUp(self):
        super(TestShareInstanceList, self).setUp()
        self.instances_list = manila_fakes.FakeShareInstance.create_share_instances(count=2)
        self.instances_mock.list.return_value = self.instances_list
        self.share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.get.return_value = self.share
        self.shares_mock.list_instances.return_value = self.instances_list
        self.shares_mock.list_instances.return_value = self.instances_list
        self.instance_values = (oscutils.get_dict_properties(instance._info, self.columns) for instance in self.instances_list)
        self.cmd = osc_share_instances.ShareInstanceList(self.app, None)

    def test_share_instance_list(self):
        argslist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, argslist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertIs(True, self.instances_mock.list.called)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.instance_values), list(data))

    def test_share_instance_list_by_share(self):
        argslist = ['--share', self.share['id']]
        verifylist = [('share', self.share.id)]
        parsed_args = self.check_parser(self.cmd, argslist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.list_instances.assert_called_with(self.share)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.instance_values), list(data))

    def test_share_instance_list_by_export_location(self):
        fake_export_location = '10.1.1.0:/fake_share_el'
        argslist = ['--export-location', fake_export_location]
        verifylist = [('export_location', fake_export_location)]
        parsed_args = self.check_parser(self.cmd, argslist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.instances_mock.list.assert_called_with(export_location=fake_export_location)
        self.assertEqual(self.column_headers, columns)
        self.assertEqual(list(self.instance_values), list(data))

    def test_share_instance_list_by_export_location_invalid_version(self):
        fake_export_location = '10.1.1.0:/fake_share_el'
        argslist = ['--export-location', fake_export_location]
        verifylist = [('export_location', fake_export_location)]
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.34')
        parsed_args = self.check_parser(self.cmd, argslist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)