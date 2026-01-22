from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateSecurityGroupNetwork(TestSecurityGroupNetwork):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    _security_group = network_fakes.FakeSecurityGroup.create_one_security_group()
    columns = ('description', 'id', 'name', 'project_id', 'rules', 'stateful', 'tags')
    data = (_security_group.description, _security_group.id, _security_group.name, _security_group.project_id, security_group.NetworkSecurityGroupRulesColumn([]), _security_group.stateful, _security_group.tags)

    def setUp(self):
        super(TestCreateSecurityGroupNetwork, self).setUp()
        self.network_client.create_security_group = mock.Mock(return_value=self._security_group)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = security_group.CreateSecurityGroup(self.app, self.namespace)

    def test_create_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_create_min_options(self):
        arglist = [self._security_group.name]
        verifylist = [('name', self._security_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group.assert_called_once_with(**{'description': self._security_group.name, 'name': self._security_group.name})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--description', self._security_group.description, '--project', self.project.name, '--project-domain', self.domain.name, '--stateful', self._security_group.name]
        verifylist = [('description', self._security_group.description), ('name', self._security_group.name), ('project', self.project.name), ('project_domain', self.domain.name), ('stateful', self._security_group.stateful)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group.assert_called_once_with(**{'description': self._security_group.description, 'stateful': self._security_group.stateful, 'name': self._security_group.name, 'project_id': self.project.id})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def _test_create_with_tag(self, add_tags=True):
        arglist = [self._security_group.name]
        if add_tags:
            arglist += ['--tag', 'red', '--tag', 'blue']
        else:
            arglist += ['--no-tag']
        verifylist = [('name', self._security_group.name)]
        if add_tags:
            verifylist.append(('tags', ['red', 'blue']))
        else:
            verifylist.append(('no_tag', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group.assert_called_once_with(**{'description': self._security_group.name, 'name': self._security_group.name})
        if add_tags:
            self.network_client.set_tags.assert_called_once_with(self._security_group, tests_utils.CompareBySet(['red', 'blue']))
        else:
            self.assertFalse(self.network_client.set_tags.called)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_create_with_tags(self):
        self._test_create_with_tag(add_tags=True)

    def test_create_with_no_tag(self):
        self._test_create_with_tag(add_tags=False)