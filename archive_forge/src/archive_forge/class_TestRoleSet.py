import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRoleSet(TestRole):

    def setUp(self):
        super(TestRoleSet, self).setUp()
        self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE), loaded=True)
        self.roles_mock.update.return_value = None
        self.cmd = role.SetRole(self.app, None)

    def test_role_set_no_options(self):
        arglist = ['--name', 'over', identity_fakes.role_name]
        verifylist = [('name', 'over'), ('role', identity_fakes.role_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'over', 'description': None, 'options': {}}
        self.roles_mock.update.assert_called_with(identity_fakes.role_id, **kwargs)
        self.assertIsNone(result)

    def test_role_set_domain_role(self):
        self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
        arglist = ['--name', 'over', '--domain', identity_fakes.domain_name, identity_fakes.ROLE_2['name']]
        verifylist = [('name', 'over'), ('domain', identity_fakes.domain_name), ('role', identity_fakes.ROLE_2['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'over', 'description': None, 'options': {}}
        self.roles_mock.update.assert_called_with(identity_fakes.ROLE_2['id'], **kwargs)
        self.assertIsNone(result)

    def test_role_set_description(self):
        self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
        arglist = ['--name', 'over', '--description', identity_fakes.role_description, identity_fakes.ROLE_2['name']]
        verifylist = [('name', 'over'), ('description', identity_fakes.role_description), ('role', identity_fakes.ROLE_2['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'over', 'description': identity_fakes.role_description, 'options': {}}
        self.roles_mock.update.assert_called_with(identity_fakes.ROLE_2['id'], **kwargs)
        self.assertIsNone(result)

    def test_role_set_with_immutable(self):
        self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
        arglist = ['--name', 'over', '--immutable', identity_fakes.ROLE_2['name']]
        verifylist = [('name', 'over'), ('immutable', True), ('role', identity_fakes.ROLE_2['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'over', 'description': None, 'options': {'immutable': True}}
        self.roles_mock.update.assert_called_with(identity_fakes.ROLE_2['id'], **kwargs)
        self.assertIsNone(result)

    def test_role_set_with_no_immutable(self):
        self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
        arglist = ['--name', 'over', '--no-immutable', identity_fakes.ROLE_2['name']]
        verifylist = [('name', 'over'), ('no_immutable', True), ('role', identity_fakes.ROLE_2['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'over', 'description': None, 'options': {'immutable': False}}
        self.roles_mock.update.assert_called_with(identity_fakes.ROLE_2['id'], **kwargs)
        self.assertIsNone(result)