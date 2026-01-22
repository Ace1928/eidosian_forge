import copy
from unittest import mock
from unittest.mock import call
import uuid
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import keypair
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestKeypairCreate(TestKeypair):

    def setUp(self):
        super().setUp()
        self.keypair = compute_fakes.create_one_keypair()
        self.columns = ('created_at', 'fingerprint', 'id', 'is_deleted', 'name', 'type', 'user_id')
        self.data = (self.keypair.created_at, self.keypair.fingerprint, self.keypair.id, self.keypair.is_deleted, self.keypair.name, self.keypair.type, self.keypair.user_id)
        self.cmd = keypair.CreateKeypair(self.app, None)
        self.compute_sdk_client.create_keypair.return_value = self.keypair

    @mock.patch.object(keypair, '_generate_keypair', return_value=keypair.Keypair('private', 'public'))
    def test_keypair_create_no_options(self, mock_generate):
        arglist = [self.keypair.name]
        verifylist = [('name', self.keypair.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, public_key=mock_generate.return_value.public_key)
        self.assertEqual({}, columns)
        self.assertEqual({}, data)

    def test_keypair_create_public_key(self):
        self.data = (self.keypair.created_at, self.keypair.fingerprint, self.keypair.id, self.keypair.is_deleted, self.keypair.name, self.keypair.type, self.keypair.user_id)
        arglist = ['--public-key', self.keypair.public_key, self.keypair.name]
        verifylist = [('public_key', self.keypair.public_key), ('name', self.keypair.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('io.open') as mock_open:
            mock_open.return_value = mock.MagicMock()
            m_file = mock_open.return_value.__enter__.return_value
            m_file.read.return_value = 'dummy'
            columns, data = self.cmd.take_action(parsed_args)
            self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, public_key=self.keypair.public_key)
            self.assertEqual(self.columns, columns)
            self.assertEqual(self.data, data)

    @mock.patch.object(keypair, '_generate_keypair', return_value=keypair.Keypair('private', 'public'))
    def test_keypair_create_private_key(self, mock_generate):
        tmp_pk_file = '/tmp/kp-file-' + uuid.uuid4().hex
        arglist = ['--private-key', tmp_pk_file, self.keypair.name]
        verifylist = [('private_key', tmp_pk_file), ('name', self.keypair.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('io.open') as mock_open:
            mock_open.return_value = mock.MagicMock()
            m_file = mock_open.return_value.__enter__.return_value
            columns, data = self.cmd.take_action(parsed_args)
            self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, public_key=mock_generate.return_value.public_key)
            mock_open.assert_called_once_with(tmp_pk_file, 'w+')
            m_file.write.assert_called_once_with(mock_generate.return_value.private_key)
            self.assertEqual(self.columns, columns)
            self.assertEqual(self.data, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_keypair_create_with_key_type(self, sm_mock):
        for key_type in ['x509', 'ssh']:
            self.compute_sdk_client.create_keypair.return_value = self.keypair
            self.data = (self.keypair.created_at, self.keypair.fingerprint, self.keypair.id, self.keypair.is_deleted, self.keypair.name, self.keypair.type, self.keypair.user_id)
            arglist = ['--public-key', self.keypair.public_key, self.keypair.name, '--type', key_type]
            verifylist = [('public_key', self.keypair.public_key), ('name', self.keypair.name), ('type', key_type)]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            with mock.patch('io.open') as mock_open:
                mock_open.return_value = mock.MagicMock()
                m_file = mock_open.return_value.__enter__.return_value
                m_file.read.return_value = 'dummy'
                columns, data = self.cmd.take_action(parsed_args)
            self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, public_key=self.keypair.public_key, key_type=key_type)
            self.assertEqual(self.columns, columns)
            self.assertEqual(self.data, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_keypair_create_with_key_type_pre_v22(self, sm_mock):
        for key_type in ['x509', 'ssh']:
            arglist = ['--public-key', self.keypair.public_key, self.keypair.name, '--type', 'ssh']
            verifylist = [('public_key', self.keypair.public_key), ('name', self.keypair.name), ('type', 'ssh')]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            with mock.patch('io.open') as mock_open:
                mock_open.return_value = mock.MagicMock()
                m_file = mock_open.return_value.__enter__.return_value
                m_file.read.return_value = 'dummy'
                ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
            self.assertIn('--os-compute-api-version 2.2 or greater is required', str(ex))

    @mock.patch.object(keypair, '_generate_keypair', return_value=keypair.Keypair('private', 'public'))
    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_key_pair_create_with_user(self, sm_mock, mock_generate):
        arglist = ['--user', identity_fakes.user_name, self.keypair.name]
        verifylist = [('user', identity_fakes.user_name), ('name', self.keypair.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.create_keypair.assert_called_with(name=self.keypair.name, user_id=identity_fakes.user_id, public_key=mock_generate.return_value.public_key)
        self.assertEqual({}, columns)
        self.assertEqual({}, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_key_pair_create_with_user_pre_v210(self, sm_mock):
        arglist = ['--user', identity_fakes.user_name, self.keypair.name]
        verifylist = [('user', identity_fakes.user_name), ('name', self.keypair.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.10 or greater is required', str(ex))