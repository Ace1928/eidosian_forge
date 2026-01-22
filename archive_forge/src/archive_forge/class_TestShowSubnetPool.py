from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestShowSubnetPool(TestSubnetPool):
    _subnet_pool = network_fakes.FakeSubnetPool.create_one_subnet_pool()
    columns = ('address_scope_id', 'default_prefixlen', 'default_quota', 'description', 'id', 'ip_version', 'is_default', 'max_prefixlen', 'min_prefixlen', 'name', 'prefixes', 'project_id', 'shared', 'tags')
    data = (_subnet_pool.address_scope_id, _subnet_pool.default_prefixlen, _subnet_pool.default_quota, _subnet_pool.description, _subnet_pool.id, _subnet_pool.ip_version, _subnet_pool.is_default, _subnet_pool.max_prefixlen, _subnet_pool.min_prefixlen, _subnet_pool.name, format_columns.ListColumn(_subnet_pool.prefixes), _subnet_pool.project_id, _subnet_pool.shared, format_columns.ListColumn(_subnet_pool.tags))

    def setUp(self):
        super(TestShowSubnetPool, self).setUp()
        self.network_client.find_subnet_pool = mock.Mock(return_value=self._subnet_pool)
        self.cmd = subnet_pool.ShowSubnetPool(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._subnet_pool.name]
        verifylist = [('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_subnet_pool.assert_called_once_with(self._subnet_pool.name, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)