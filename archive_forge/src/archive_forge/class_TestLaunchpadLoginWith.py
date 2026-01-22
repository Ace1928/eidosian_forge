from contextlib import contextmanager
import os
import shutil
import socket
import stat
import tempfile
import unittest
import warnings
from lazr.restfulclient.resource import ServiceRoot
from launchpadlib.credentials import (
from launchpadlib import uris
import launchpadlib.launchpad
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class TestLaunchpadLoginWith(KeyringTest):
    """Tests for Launchpad.login_with()."""

    def setUp(self):
        super(TestLaunchpadLoginWith, self).setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super(TestLaunchpadLoginWith, self).tearDown()
        shutil.rmtree(self.temp_dir)

    def test_dirs_created(self):
        launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
        NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
        self.assertTrue(os.path.isdir(launchpadlib_dir))
        service_path = os.path.join(launchpadlib_dir, 'api.example.com')
        self.assertTrue(os.path.isdir(service_path))
        self.assertTrue(os.path.isdir(os.path.join(service_path, 'cache')))
        credentials_path = os.path.join(service_path, 'credentials')
        self.assertFalse(os.path.isdir(credentials_path))

    def test_dirs_created_are_changed_to_secure(self):
        launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
        os.mkdir(launchpadlib_dir)
        os.chmod(launchpadlib_dir, 493)
        self.assertTrue(os.path.isdir(launchpadlib_dir))
        statinfo = os.stat(launchpadlib_dir)
        mode = stat.S_IMODE(statinfo.st_mode)
        self.assertNotEqual(mode, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
        statinfo = os.stat(launchpadlib_dir)
        mode = stat.S_IMODE(statinfo.st_mode)
        self.assertEqual(mode, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)

    def test_dirs_created_are_secure(self):
        launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
        NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
        self.assertTrue(os.path.isdir(launchpadlib_dir))
        statinfo = os.stat(launchpadlib_dir)
        mode = stat.S_IMODE(statinfo.st_mode)
        self.assertEqual(mode, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)

    def test_version_is_propagated(self):
        launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
        launchpad = NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir, version='foo')
        self.assertEqual(launchpad.passed_in_args['version'], 'foo')
        launchpad = NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir, version='bar')
        self.assertEqual(launchpad.passed_in_args['version'], 'bar')

    def test_application_name_is_propagated(self):
        launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
        launchpad = NoNetworkLaunchpad.login_with('very important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
        self.assertEqual(launchpad.credentials.consumer.application_name, 'very important')
        launchpad = NoNetworkLaunchpad.login_with('very important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
        self.assertEqual(launchpad.credentials.consumer.application_name, 'very important')

    def test_authorization_engine_is_propagated(self):
        engine = NoNetworkAuthorizationEngine(SERVICE_ROOT, 'application name')
        NoNetworkLaunchpad.login_with(authorization_engine=engine)
        self.assertEqual(engine.request_tokens_obtained, 1)
        self.assertEqual(engine.access_tokens_obtained, 1)

    def test_login_with_must_identify_application(self):
        self.assertRaises(ValueError, NoNetworkLaunchpad.login_with)

    def test_application_name_identifies_app(self):
        NoNetworkLaunchpad.login_with(application_name='name')

    def test_consumer_name_identifies_app(self):
        NoNetworkLaunchpad.login_with(consumer_name='name')

    def test_inconsistent_application_name_rejected(self):
        """Catch an attempt to specify inconsistent application_names."""
        engine = NoNetworkAuthorizationEngine(SERVICE_ROOT, 'application name1')
        self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, 'application name2', authorization_engine=engine)

    def test_inconsistent_consumer_name_rejected(self):
        """Catch an attempt to specify inconsistent application_names."""
        engine = NoNetworkAuthorizationEngine(SERVICE_ROOT, None, consumer_name='consumer_name1')
        self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, 'consumer_name2', authorization_engine=engine)

    def test_inconsistent_allow_access_levels_rejected(self):
        """Catch an attempt to specify inconsistent allow_access_levels."""
        engine = NoNetworkAuthorizationEngine(SERVICE_ROOT, consumer_name='consumer', allow_access_levels=['FOO'])
        self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, None, consumer_name='consumer', allow_access_levels=['BAR'], authorization_engine=engine)

    def test_inconsistent_credential_save_failed(self):

        def callback1():
            pass
        store = KeyringCredentialStore(credential_save_failed=callback1)

        def callback2():
            pass
        self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, 'app name', credential_store=store, credential_save_failed=callback2)

    def test_non_desktop_integration(self):
        launchpad = NoNetworkLaunchpad.login_with(consumer_name='consumer', allow_access_levels=['FOO'])
        self.assertEqual(launchpad.credentials.consumer.key, 'consumer')
        self.assertEqual(launchpad.credentials.consumer.application_name, None)
        self.assertEqual(launchpad.authorization_engine.allow_access_levels, ['FOO'])

    def test_desktop_integration_doesnt_happen_without_consumer_name(self):
        launchpad = NoNetworkLaunchpad.login_with('application name', allow_access_levels=['FOO'])
        self.assertEqual(launchpad.authorization_engine.allow_access_levels, ['DESKTOP_INTEGRATION'])

    def test_no_credentials_creates_new_credential(self):
        timeout = object()
        proxy_info = object()
        launchpad = NoNetworkLaunchpad.login_with('app name', launchpadlib_dir=self.temp_dir, service_root=SERVICE_ROOT, timeout=timeout, proxy_info=proxy_info)
        self.assertEqual(launchpad.credentials.access_token.key, NoNetworkAuthorizationEngine.ACCESS_TOKEN_KEY)
        self.assertEqual(launchpad.credentials.consumer.application_name, 'app name')
        self.assertEqual(launchpad.authorization_engine.allow_access_levels, ['DESKTOP_INTEGRATION'])
        expected_arguments = dict(service_root=SERVICE_ROOT, cache=os.path.join(self.temp_dir, 'api.example.com', 'cache'), timeout=timeout, proxy_info=proxy_info, version=NoNetworkLaunchpad.DEFAULT_VERSION)
        self.assertEqual(launchpad.passed_in_args, expected_arguments)

    def test_anonymous_login(self):
        """Test the anonymous login helper function."""
        launchpad = NoNetworkLaunchpad.login_anonymously('anonymous access', launchpadlib_dir=self.temp_dir, service_root=SERVICE_ROOT)
        self.assertEqual(launchpad.credentials.access_token.key, '')
        self.assertEqual(launchpad.credentials.access_token.secret, '')
        credentials_path = os.path.join(self.temp_dir, 'api.example.com', 'credentials', 'anonymous access')
        self.assertFalse(os.path.exists(credentials_path))

    def test_existing_credentials_arguments_passed_on(self):
        os.makedirs(os.path.join(self.temp_dir, 'api.example.com', 'credentials'))
        credentials_file_path = os.path.join(self.temp_dir, 'api.example.com', 'credentials', 'app name')
        credentials = Credentials('app name', consumer_secret='consumer_secret:42', access_token=AccessToken('access_key:84', 'access_secret:168'))
        credentials.save_to_path(credentials_file_path)
        timeout = object()
        proxy_info = object()
        version = 'foo'
        launchpad = NoNetworkLaunchpad.login_with('app name', launchpadlib_dir=self.temp_dir, service_root=SERVICE_ROOT, timeout=timeout, proxy_info=proxy_info, version=version)
        expected_arguments = dict(service_root=SERVICE_ROOT, timeout=timeout, proxy_info=proxy_info, version=version, cache=os.path.join(self.temp_dir, 'api.example.com', 'cache'))
        for key, expected in expected_arguments.items():
            actual = launchpad.passed_in_args[key]
            self.assertEqual(actual, expected)

    def test_None_launchpadlib_dir(self):
        old_home = os.environ.get('HOME')
        os.environ['HOME'] = self.temp_dir
        launchpad = NoNetworkLaunchpad.login_with('app name', service_root=SERVICE_ROOT)
        if old_home is not None:
            os.environ['HOME'] = old_home
        else:
            del os.environ['HOME']
        cache_dir = launchpad.passed_in_args['cache']
        launchpadlib_dir = os.path.abspath(os.path.join(cache_dir, '..', '..'))
        self.assertEqual(launchpadlib_dir, os.path.join(self.temp_dir, '.launchpadlib'))
        self.assertTrue(os.path.exists(os.path.join(launchpadlib_dir, 'api.example.com', 'cache')))

    def test_short_service_name(self):
        launchpad = NoNetworkLaunchpad.login_with('app name', 'staging')
        self.assertEqual(launchpad.passed_in_args['service_root'], 'https://api.staging.launchpad.net/')
        launchpad = NoNetworkLaunchpad.login_with('app name', uris.service_roots['staging'])
        self.assertEqual(launchpad.passed_in_args['service_root'], uris.service_roots['staging'])
        launchpad = ('app name', 'https://')
        self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, 'app name', 'foo')

    def test_max_failed_attempts_accepted(self):
        NoNetworkLaunchpad.login_with('not important', max_failed_attempts=5)