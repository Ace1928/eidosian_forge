from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowAddressGroup(TestAddressGroup):
    _address_group = network_fakes.create_one_address_group()
    columns = ('addresses', 'description', 'id', 'name', 'project_id')
    data = (_address_group.addresses, _address_group.description, _address_group.id, _address_group.name, _address_group.project_id)

    def setUp(self):
        super(TestShowAddressGroup, self).setUp()
        self.network_client.find_address_group = mock.Mock(return_value=self._address_group)
        self.cmd = address_group.ShowAddressGroup(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._address_group.name]
        verifylist = [('address_group', self._address_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_address_group.assert_called_once_with(self._address_group.name, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))