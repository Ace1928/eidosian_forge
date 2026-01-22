from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestCreateSubnetPool(TestSubnetPool):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    _subnet_pool = network_fakes.FakeSubnetPool.create_one_subnet_pool()
    _address_scope = network_fakes.create_one_address_scope()
    columns = ('address_scope_id', 'default_prefixlen', 'default_quota', 'description', 'id', 'ip_version', 'is_default', 'max_prefixlen', 'min_prefixlen', 'name', 'prefixes', 'project_id', 'shared', 'tags')
    data = (_subnet_pool.address_scope_id, _subnet_pool.default_prefixlen, _subnet_pool.default_quota, _subnet_pool.description, _subnet_pool.id, _subnet_pool.ip_version, _subnet_pool.is_default, _subnet_pool.max_prefixlen, _subnet_pool.min_prefixlen, _subnet_pool.name, format_columns.ListColumn(_subnet_pool.prefixes), _subnet_pool.project_id, _subnet_pool.shared, format_columns.ListColumn(_subnet_pool.tags))

    def setUp(self):
        super(TestCreateSubnetPool, self).setUp()
        self.network_client.create_subnet_pool = mock.Mock(return_value=self._subnet_pool)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = subnet_pool.CreateSubnetPool(self.app, self.namespace)
        self.network_client.find_address_scope = mock.Mock(return_value=self._address_scope)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
        self.assertFalse(self.network_client.set_tags.called)

    def test_create_no_pool_prefix(self):
        """Make sure --pool-prefix is a required argument"""
        arglist = [self._subnet_pool.name]
        verifylist = [('name', self._subnet_pool.name)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = ['--pool-prefix', '10.0.10.0/24', self._subnet_pool.name]
        verifylist = [('prefixes', ['10.0.10.0/24']), ('name', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_subnet_pool.assert_called_once_with(**{'prefixes': ['10.0.10.0/24'], 'name': self._subnet_pool.name})
        self.assertFalse(self.network_client.set_tags.called)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_prefixlen_options(self):
        arglist = ['--default-prefix-length', self._subnet_pool.default_prefixlen, '--max-prefix-length', self._subnet_pool.max_prefixlen, '--min-prefix-length', self._subnet_pool.min_prefixlen, '--pool-prefix', '10.0.10.0/24', self._subnet_pool.name]
        verifylist = [('default_prefix_length', int(self._subnet_pool.default_prefixlen)), ('max_prefix_length', int(self._subnet_pool.max_prefixlen)), ('min_prefix_length', int(self._subnet_pool.min_prefixlen)), ('name', self._subnet_pool.name), ('prefixes', ['10.0.10.0/24'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_subnet_pool.assert_called_once_with(**{'default_prefixlen': int(self._subnet_pool.default_prefixlen), 'max_prefixlen': int(self._subnet_pool.max_prefixlen), 'min_prefixlen': int(self._subnet_pool.min_prefixlen), 'prefixes': ['10.0.10.0/24'], 'name': self._subnet_pool.name})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_len_negative(self):
        arglist = [self._subnet_pool.name, '--min-prefix-length', '-16']
        verifylist = [('subnet_pool', self._subnet_pool.name), ('min_prefix_length', '-16')]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_project_domain(self):
        arglist = ['--pool-prefix', '10.0.10.0/24', '--project', self.project.name, '--project-domain', self.domain.name, self._subnet_pool.name]
        verifylist = [('prefixes', ['10.0.10.0/24']), ('project', self.project.name), ('project_domain', self.domain.name), ('name', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_subnet_pool.assert_called_once_with(**{'prefixes': ['10.0.10.0/24'], 'project_id': self.project.id, 'name': self._subnet_pool.name})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_address_scope_option(self):
        arglist = ['--pool-prefix', '10.0.10.0/24', '--address-scope', self._address_scope.id, self._subnet_pool.name]
        verifylist = [('prefixes', ['10.0.10.0/24']), ('address_scope', self._address_scope.id), ('name', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_subnet_pool.assert_called_once_with(**{'prefixes': ['10.0.10.0/24'], 'address_scope_id': self._address_scope.id, 'name': self._subnet_pool.name})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_default_and_shared_options(self):
        arglist = ['--pool-prefix', '10.0.10.0/24', '--default', '--share', self._subnet_pool.name]
        verifylist = [('prefixes', ['10.0.10.0/24']), ('default', True), ('share', True), ('name', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_subnet_pool.assert_called_once_with(**{'is_default': True, 'name': self._subnet_pool.name, 'prefixes': ['10.0.10.0/24'], 'shared': True})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_description(self):
        arglist = ['--pool-prefix', '10.0.10.0/24', '--description', self._subnet_pool.description, self._subnet_pool.name]
        verifylist = [('prefixes', ['10.0.10.0/24']), ('description', self._subnet_pool.description), ('name', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_subnet_pool.assert_called_once_with(**{'name': self._subnet_pool.name, 'prefixes': ['10.0.10.0/24'], 'description': self._subnet_pool.description})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_default_quota(self):
        arglist = ['--pool-prefix', '10.0.10.0/24', '--default-quota', '10', self._subnet_pool.name]
        verifylist = [('prefixes', ['10.0.10.0/24']), ('default_quota', 10), ('name', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_subnet_pool.assert_called_once_with(**{'name': self._subnet_pool.name, 'prefixes': ['10.0.10.0/24'], 'default_quota': 10})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def _test_create_with_tag(self, add_tags=True):
        arglist = ['--pool-prefix', '10.0.10.0/24', self._subnet_pool.name]
        if add_tags:
            arglist += ['--tag', 'red', '--tag', 'blue']
        else:
            arglist += ['--no-tag']
        verifylist = [('prefixes', ['10.0.10.0/24']), ('name', self._subnet_pool.name)]
        if add_tags:
            verifylist.append(('tags', ['red', 'blue']))
        else:
            verifylist.append(('no_tag', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_subnet_pool.assert_called_once_with(prefixes=['10.0.10.0/24'], name=self._subnet_pool.name)
        if add_tags:
            self.network_client.set_tags.assert_called_once_with(self._subnet_pool, test_utils.CompareBySet(['red', 'blue']))
        else:
            self.assertFalse(self.network_client.set_tags.called)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_tags(self):
        self._test_create_with_tag(add_tags=True)

    def test_create_with_no_tag(self):
        self._test_create_with_tag(add_tags=False)