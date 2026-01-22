from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestListSubnetPool(TestSubnetPool):
    _subnet_pools = network_fakes.FakeSubnetPool.create_subnet_pools(count=3)
    columns = ('ID', 'Name', 'Prefixes')
    columns_long = columns + ('Default Prefix Length', 'Address Scope', 'Default Subnet Pool', 'Shared', 'Tags')
    data = []
    for pool in _subnet_pools:
        data.append((pool.id, pool.name, format_columns.ListColumn(pool.prefixes)))
    data_long = []
    for pool in _subnet_pools:
        data_long.append((pool.id, pool.name, format_columns.ListColumn(pool.prefixes), pool.default_prefixlen, pool.address_scope_id, pool.is_default, pool.shared, format_columns.ListColumn(pool.tags)))

    def setUp(self):
        super(TestListSubnetPool, self).setUp()
        self.cmd = subnet_pool.ListSubnetPool(self.app, self.namespace)
        self.network_client.subnet_pools = mock.Mock(return_value=self._subnet_pools)

    def test_subnet_pool_list_no_option(self):
        arglist = []
        verifylist = [('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.subnet_pools.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_pool_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.subnet_pools.assert_called_once_with()
        self.assertEqual(self.columns_long, columns)
        self.assertCountEqual(self.data_long, list(data))

    def test_subnet_pool_list_no_share(self):
        arglist = ['--no-share']
        verifylist = [('share', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'shared': False, 'is_shared': False}
        self.network_client.subnet_pools.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_pool_list_share(self):
        arglist = ['--share']
        verifylist = [('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'shared': True, 'is_shared': True}
        self.network_client.subnet_pools.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_pool_list_no_default(self):
        arglist = ['--no-default']
        verifylist = [('default', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'is_default': False}
        self.network_client.subnet_pools.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_pool_list_default(self):
        arglist = ['--default']
        verifylist = [('default', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'is_default': True}
        self.network_client.subnet_pools.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_pool_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.subnet_pools.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_pool_list_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id), ('project_domain', project.domain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.subnet_pools.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_pool_list_name(self):
        subnet_pool = network_fakes.FakeSubnetPool.create_one_subnet_pool()
        self.network_client.find_network = mock.Mock(return_value=subnet_pool)
        arglist = ['--name', subnet_pool.name]
        verifylist = [('name', subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'name': subnet_pool.name}
        self.network_client.subnet_pools.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_pool_list_address_scope(self):
        addr_scope = network_fakes.create_one_address_scope()
        self.network_client.find_address_scope = mock.Mock(return_value=addr_scope)
        arglist = ['--address-scope', addr_scope.id]
        verifylist = [('address_scope', addr_scope.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'address_scope_id': addr_scope.id}
        self.network_client.subnet_pools.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_with_tag_options(self):
        arglist = ['--tags', 'red,blue', '--any-tags', 'red,green', '--not-tags', 'orange,yellow', '--not-any-tags', 'black,white']
        verifylist = [('tags', ['red', 'blue']), ('any_tags', ['red', 'green']), ('not_tags', ['orange', 'yellow']), ('not_any_tags', ['black', 'white'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.subnet_pools.assert_called_once_with(**{'tags': 'red,blue', 'any_tags': 'red,green', 'not_tags': 'orange,yellow', 'not_any_tags': 'black,white'})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))