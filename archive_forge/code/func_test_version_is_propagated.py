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
def test_version_is_propagated(self):
    launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
    launchpad = NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir, version='foo')
    self.assertEqual(launchpad.passed_in_args['version'], 'foo')
    launchpad = NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir, version='bar')
    self.assertEqual(launchpad.passed_in_args['version'], 'bar')