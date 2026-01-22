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
class TestResourceTypeClasses(unittest.TestCase):
    """launchpadlib must know about restfulclient's resource types."""

    def test_resource_types(self):
        for name, cls in ServiceRoot.RESOURCE_TYPE_CLASSES.items():
            self.assertEqual(Launchpad.RESOURCE_TYPE_CLASSES[name], cls)