from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestSetSubnetPool(TestSubnetPool):
    _subnet_pool = network_fakes.FakeSubnetPool.create_one_subnet_pool({'default_quota': 10, 'tags': ['green', 'red']})
    _address_scope = network_fakes.create_one_address_scope()

    def setUp(self):
        super(TestSetSubnetPool, self).setUp()
        self.network_client.update_subnet_pool = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.network_client.find_subnet_pool = mock.Mock(return_value=self._subnet_pool)
        self.network_client.find_address_scope = mock.Mock(return_value=self._address_scope)
        self.cmd = subnet_pool.SetSubnetPool(self.app, self.namespace)

    def test_set_this(self):
        arglist = ['--name', 'noob', '--default-prefix-length', '8', '--min-prefix-length', '8', self._subnet_pool.name]
        verifylist = [('name', 'noob'), ('default_prefix_length', 8), ('min_prefix_length', 8), ('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'noob', 'default_prefixlen': 8, 'min_prefixlen': 8}
        self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **attrs)
        self.assertIsNone(result)

    def test_set_that(self):
        arglist = ['--pool-prefix', '10.0.1.0/24', '--pool-prefix', '10.0.2.0/24', '--max-prefix-length', '16', self._subnet_pool.name]
        verifylist = [('prefixes', ['10.0.1.0/24', '10.0.2.0/24']), ('max_prefix_length', 16), ('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        prefixes = ['10.0.1.0/24', '10.0.2.0/24']
        prefixes.extend(self._subnet_pool.prefixes)
        attrs = {'prefixes': prefixes, 'max_prefixlen': 16}
        self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **attrs)
        self.assertIsNone(result)

    def test_set_nothing(self):
        arglist = [self._subnet_pool.name]
        verifylist = [('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_subnet_pool.called)
        self.assertFalse(self.network_client.set_tags.called)
        self.assertIsNone(result)

    def test_set_len_negative(self):
        arglist = ['--max-prefix-length', '-16', self._subnet_pool.name]
        verifylist = [('max_prefix_length', '-16'), ('subnet_pool', self._subnet_pool.name)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_address_scope(self):
        arglist = ['--address-scope', self._address_scope.id, self._subnet_pool.name]
        verifylist = [('address_scope', self._address_scope.id), ('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'address_scope_id': self._address_scope.id}
        self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **attrs)
        self.assertIsNone(result)

    def test_set_no_address_scope(self):
        arglist = ['--no-address-scope', self._subnet_pool.name]
        verifylist = [('no_address_scope', True), ('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'address_scope_id': None}
        self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **attrs)
        self.assertIsNone(result)

    def test_set_no_address_scope_conflict(self):
        arglist = ['--address-scope', self._address_scope.id, '--no-address-scope', self._subnet_pool.name]
        verifylist = [('address_scope', self._address_scope.id), ('no_address_scope', True), ('subnet_pool', self._subnet_pool.name)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_default(self):
        arglist = ['--default', self._subnet_pool.name]
        verifylist = [('default', True), ('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'is_default': True}
        self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **attrs)
        self.assertIsNone(result)

    def test_set_no_default(self):
        arglist = ['--no-default', self._subnet_pool.name]
        verifylist = [('no_default', True), ('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'is_default': False}
        self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **attrs)
        self.assertIsNone(result)

    def test_set_no_default_conflict(self):
        arglist = ['--default', '--no-default', self._subnet_pool.name]
        verifylist = [('default', True), ('no_default', True), ('subnet_pool', self._subnet_pool.name)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_description(self):
        arglist = ['--description', 'new_description', self._subnet_pool.name]
        verifylist = [('description', 'new_description'), ('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'description': 'new_description'}
        self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **attrs)
        self.assertIsNone(result)

    def test_set_with_default_quota(self):
        arglist = ['--default-quota', '20', self._subnet_pool.name]
        verifylist = [('default_quota', 20), ('subnet_pool', self._subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_subnet_pool.assert_called_once_with(self._subnet_pool, **{'default_quota': 20})
        self.assertIsNone(result)

    def _test_set_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['red', 'blue', 'green']
        else:
            arglist = ['--no-tag']
            verifylist = [('no_tag', True)]
            expected_args = []
        arglist.append(self._subnet_pool.name)
        verifylist.append(('subnet_pool', self._subnet_pool.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_subnet_pool.called)
        self.network_client.set_tags.assert_called_once_with(self._subnet_pool, test_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_set_with_tags(self):
        self._test_set_tags(with_tags=True)

    def test_set_with_no_tag(self):
        self._test_set_tags(with_tags=False)