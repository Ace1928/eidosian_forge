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