from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowAddressScope(TestAddressScope):
    _address_scope = network_fakes.create_one_address_scope()
    columns = ('id', 'ip_version', 'name', 'project_id', 'shared')
    data = (_address_scope.id, _address_scope.ip_version, _address_scope.name, _address_scope.project_id, _address_scope.is_shared)

    def setUp(self):
        super(TestShowAddressScope, self).setUp()
        self.network_client.find_address_scope = mock.Mock(return_value=self._address_scope)
        self.cmd = address_scope.ShowAddressScope(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._address_scope.name]
        verifylist = [('address_scope', self._address_scope.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_address_scope.assert_called_once_with(self._address_scope.name, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.data), list(data))