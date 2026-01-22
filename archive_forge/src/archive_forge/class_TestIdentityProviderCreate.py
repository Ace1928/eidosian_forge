import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
class TestIdentityProviderCreate(TestIdentityProvider):
    columns = ('description', 'domain_id', 'enabled', 'id', 'remote_ids')
    datalist = (identity_fakes.idp_description, identity_fakes.domain_id, True, identity_fakes.idp_id, identity_fakes.formatted_idp_remote_ids)

    def setUp(self):
        super(TestIdentityProviderCreate, self).setUp()
        copied_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
        resource = fakes.FakeResource(None, copied_idp, loaded=True)
        self.identity_providers_mock.create.return_value = resource
        self.cmd = identity_provider.CreateIdentityProvider(self.app, None)

    def test_create_identity_provider_no_options(self):
        arglist = [identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': None, 'enabled': True, 'description': None, 'domain_id': None}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_description(self):
        arglist = ['--description', identity_fakes.idp_description, identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('description', identity_fakes.idp_description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': None, 'description': identity_fakes.idp_description, 'domain_id': None, 'enabled': True}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_remote_id(self):
        arglist = [identity_fakes.idp_id, '--remote-id', identity_fakes.idp_remote_ids[0]]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('remote_id', identity_fakes.idp_remote_ids[:1])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': identity_fakes.idp_remote_ids[:1], 'description': None, 'domain_id': None, 'enabled': True}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_remote_ids_multiple(self):
        arglist = ['--remote-id', identity_fakes.idp_remote_ids[0], '--remote-id', identity_fakes.idp_remote_ids[1], identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('remote_id', identity_fakes.idp_remote_ids)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': identity_fakes.idp_remote_ids, 'description': None, 'domain_id': None, 'enabled': True}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_remote_ids_file(self):
        arglist = ['--remote-id-file', '/tmp/file_name', identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('remote_id_file', '/tmp/file_name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        mocker = mock.Mock()
        mocker.return_value = '\n'.join(identity_fakes.idp_remote_ids)
        with mock.patch('openstackclient.identity.v3.identity_provider.utils.read_blob_file_contents', mocker):
            columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': identity_fakes.idp_remote_ids, 'description': None, 'domain_id': None, 'enabled': True}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_disabled(self):
        IDENTITY_PROVIDER = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
        IDENTITY_PROVIDER['enabled'] = False
        IDENTITY_PROVIDER['description'] = None
        resource = fakes.FakeResource(None, IDENTITY_PROVIDER, loaded=True)
        self.identity_providers_mock.create.return_value = resource
        arglist = ['--disable', identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': None, 'enabled': False, 'description': None, 'domain_id': None}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        datalist = (None, identity_fakes.domain_id, False, identity_fakes.idp_id, identity_fakes.formatted_idp_remote_ids)
        self.assertCountEqual(datalist, data)

    def test_create_identity_provider_domain_name(self):
        arglist = ['--domain', identity_fakes.domain_name, identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('domain', identity_fakes.domain_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': None, 'description': None, 'domain_id': identity_fakes.domain_id, 'enabled': True}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_domain_id(self):
        arglist = ['--domain', identity_fakes.domain_id, identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('domain', identity_fakes.domain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': None, 'description': None, 'domain_id': identity_fakes.domain_id, 'enabled': True}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_authttl_positive(self):
        arglist = ['--authorization-ttl', '60', identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('authorization_ttl', 60)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': None, 'description': None, 'domain_id': None, 'enabled': True, 'authorization_ttl': 60}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_authttl_zero(self):
        arglist = ['--authorization-ttl', '0', identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('authorization_ttl', 0)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'remote_ids': None, 'description': None, 'domain_id': None, 'enabled': True, 'authorization_ttl': 0}
        self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_create_identity_provider_authttl_negative(self):
        arglist = ['--authorization-ttl', '-60', identity_fakes.idp_id]
        verifylist = [('identity_provider_id', identity_fakes.idp_id), ('authorization_ttl', -60)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_identity_provider_authttl_not_int(self):
        arglist = ['--authorization-ttl', 'spam', identity_fakes.idp_id]
        verifylist = []
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)