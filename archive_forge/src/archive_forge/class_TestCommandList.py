from unittest import mock
from openstackclient.common import module as osc_module
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
class TestCommandList(utils.TestCommand):

    def setUp(self):
        super(TestCommandList, self).setUp()
        self.app.command_manager = mock.Mock()
        self.app.command_manager.get_command_groups.return_value = ['openstack.common']
        self.app.command_manager.get_command_names.return_value = ['limits show\nextension list']
        self.cmd = osc_module.ListCommand(self.app, None)

    def test_command_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('Command Group', 'Commands')
        self.assertEqual(collist, columns)
        datalist = (('openstack.common', 'limits show\nextension list'),)
        self.assertEqual(datalist, tuple(data))

    def test_command_list_with_group_not_found(self):
        arglist = ['--group', 'not_exist']
        verifylist = [('group', 'not_exist')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('Command Group', 'Commands')
        self.assertEqual(collist, columns)
        self.assertEqual([], data)

    def test_command_list_with_group(self):
        arglist = ['--group', 'common']
        verifylist = [('group', 'common')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = ('Command Group', 'Commands')
        self.assertEqual(collist, columns)
        datalist = (('openstack.common', 'limits show\nextension list'),)
        self.assertEqual(datalist, tuple(data))