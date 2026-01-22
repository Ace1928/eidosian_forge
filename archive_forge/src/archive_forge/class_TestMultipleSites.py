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
class TestMultipleSites(unittest.TestCase):

    def setUp(self):
        launchpadlib.launchpad.socket = FauxSocketModule()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        launchpadlib.launchpad.socket = socket
        shutil.rmtree(self.temp_dir)

    @patch.object(NoNetworkLaunchpad, '_is_sudo', staticmethod(lambda: False))
    def test_components_of_application_key(self):
        launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
        keyring = InMemoryKeyring()
        service_root = 'http://api.example.com/'
        application_name = 'Super App 3000'
        with fake_keyring(keyring):
            launchpad = NoNetworkLaunchpad.login_with(application_name, service_root=service_root, launchpadlib_dir=launchpadlib_dir)
            consumer_name = launchpad.credentials.consumer.key
        application_key = list(keyring.data.keys())[0][1]
        self.assertIn(service_root, application_key)
        self.assertIn(consumer_name, application_key)
        self.assertEqual(application_key, consumer_name + '@' + service_root)

    @patch.object(NoNetworkLaunchpad, '_is_sudo', staticmethod(lambda: False))
    def test_same_app_different_servers(self):
        launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
        keyring = InMemoryKeyring()
        assert not keyring.data, 'oops, a fresh keyring has data in it'
        with fake_keyring(keyring):
            NoNetworkLaunchpad.login_with('application name', service_root='http://alpha.example.com/', launchpadlib_dir=launchpadlib_dir)
            NoNetworkLaunchpad.login_with('application name', service_root='http://beta.example.com/', launchpadlib_dir=launchpadlib_dir)
        assert len(keyring.data.keys()) == 2
        application_key_1 = list(keyring.data.keys())[0][1]
        application_key_2 = list(keyring.data.keys())[1][1]
        self.assertNotEqual(application_key_1, application_key_2)