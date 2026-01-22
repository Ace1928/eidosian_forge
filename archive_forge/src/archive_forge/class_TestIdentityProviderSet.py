import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
class TestIdentityProviderSet(TestIdentityProvider):
    columns = ('description', 'enabled', 'id', 'remote_ids')
    datalist = (identity_fakes.idp_description, True, identity_fakes.idp_id, identity_fakes.idp_remote_ids)

    def setUp(self):
        super(TestIdentityProviderSet, self).setUp()
        self.cmd = identity_provider.SetIdentityProvider(self.app, None)

    def test_identity_provider_set_description(self):
        """Set Identity Provider's description."""

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
            updated_idp['enabled'] = False
            resources = fakes.FakeResource(None, updated_idp, loaded=True)
            self.identity_providers_mock.update.return_value = resources
        prepare(self)
        new_description = 'new desc'
        arglist = ['--description', new_description, identity_fakes.idp_id]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('description', new_description), ('enable', False), ('disable', False), ('remote_id', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, description=new_description)

    def test_identity_provider_disable(self):
        """Disable Identity Provider

        Set Identity Provider's ``enabled`` attribute to False.
        """

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
            updated_idp['enabled'] = False
            resources = fakes.FakeResource(None, updated_idp, loaded=True)
            self.identity_providers_mock.update.return_value = resources
        prepare(self)
        arglist = ['--disable', identity_fakes.idp_id, '--remote-id', identity_fakes.idp_remote_ids[0], '--remote-id', identity_fakes.idp_remote_ids[1]]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('description', None), ('enable', False), ('disable', True), ('remote_id', identity_fakes.idp_remote_ids)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, enabled=False, remote_ids=identity_fakes.idp_remote_ids)

    def test_identity_provider_enable(self):
        """Enable Identity Provider.

        Set Identity Provider's ``enabled`` attribute to True.
        """

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            resources = fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)
            self.identity_providers_mock.update.return_value = resources
        prepare(self)
        arglist = ['--enable', identity_fakes.idp_id, '--remote-id', identity_fakes.idp_remote_ids[0], '--remote-id', identity_fakes.idp_remote_ids[1]]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('description', None), ('enable', True), ('disable', False), ('remote_id', identity_fakes.idp_remote_ids)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, enabled=True, remote_ids=identity_fakes.idp_remote_ids)

    def test_identity_provider_replace_remote_ids(self):
        """Enable Identity Provider.

        Set Identity Provider's ``enabled`` attribute to True.
        """

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            self.new_remote_id = 'new_entity'
            updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
            updated_idp['remote_ids'] = [self.new_remote_id]
            resources = fakes.FakeResource(None, updated_idp, loaded=True)
            self.identity_providers_mock.update.return_value = resources
        prepare(self)
        arglist = ['--enable', identity_fakes.idp_id, '--remote-id', self.new_remote_id]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('description', None), ('enable', True), ('disable', False), ('remote_id', [self.new_remote_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, enabled=True, remote_ids=[self.new_remote_id])

    def test_identity_provider_replace_remote_ids_file(self):
        """Enable Identity Provider.

        Set Identity Provider's ``enabled`` attribute to True.
        """

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            self.new_remote_id = 'new_entity'
            updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
            updated_idp['remote_ids'] = [self.new_remote_id]
            resources = fakes.FakeResource(None, updated_idp, loaded=True)
            self.identity_providers_mock.update.return_value = resources
        prepare(self)
        arglist = ['--enable', identity_fakes.idp_id, '--remote-id-file', self.new_remote_id]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('description', None), ('enable', True), ('disable', False), ('remote_id_file', self.new_remote_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        mocker = mock.Mock()
        mocker.return_value = self.new_remote_id
        with mock.patch('openstackclient.identity.v3.identity_provider.utils.read_blob_file_contents', mocker):
            self.cmd.take_action(parsed_args)
        self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, enabled=True, remote_ids=[self.new_remote_id])

    def test_identity_provider_no_options(self):

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            resources = fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)
            self.identity_providers_mock.get.return_value = resources
            resources = fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)
            self.identity_providers_mock.update.return_value = resources
        prepare(self)
        arglist = [identity_fakes.idp_id]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('enable', False), ('disable', False), ('remote_id', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)

    def test_identity_provider_set_authttl_positive(self):

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
            updated_idp['authorization_ttl'] = 60
            resources = fakes.FakeResource(None, updated_idp, loaded=True)
            self.identity_providers_mock.update.return_value = resources
        prepare(self)
        arglist = ['--authorization-ttl', '60', identity_fakes.idp_id]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('enable', False), ('disable', False), ('remote_id', None), ('authorization_ttl', 60)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, authorization_ttl=60)

    def test_identity_provider_set_authttl_zero(self):

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
            updated_idp['authorization_ttl'] = 0
            resources = fakes.FakeResource(None, updated_idp, loaded=True)
            self.identity_providers_mock.update.return_value = resources
        prepare(self)
        arglist = ['--authorization-ttl', '0', identity_fakes.idp_id]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('enable', False), ('disable', False), ('remote_id', None), ('authorization_ttl', 0)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, authorization_ttl=0)

    def test_identity_provider_set_authttl_negative(self):
        arglist = ['--authorization-ttl', '-1', identity_fakes.idp_id]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('enable', False), ('disable', False), ('remote_id', None), ('authorization_ttl', -1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_identity_provider_set_authttl_not_int(self):
        arglist = ['--authorization-ttl', 'spam', identity_fakes.idp_id]
        verifylist = []
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)