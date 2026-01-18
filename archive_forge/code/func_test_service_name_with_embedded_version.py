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
def test_service_name_with_embedded_version(self):
    version = 'version-foo'
    root = uris.service_roots['staging'] + version
    try:
        Launchpad(None, None, None, service_root=root, version=version)
    except ValueError as e:
        self.assertTrue(str(e).startswith('It looks like you\'re using a service root that incorporates the name of the web service version ("version-foo")'))
    else:
        raise AssertionError('Expected a ValueError that was not thrown!')
    root += '/'
    self.assertRaises(ValueError, Launchpad, None, None, None, service_root=root, version=version)
    default_version = NoNetworkLaunchpad.DEFAULT_VERSION
    root = uris.service_roots['staging'] + default_version + '/'
    self.assertRaises(ValueError, Launchpad, None, None, None, service_root=root)