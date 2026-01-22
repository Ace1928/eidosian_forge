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
class TestNameLookups(unittest.TestCase):
    """Test the utility functions in the 'uris' module."""

    def setUp(self):
        self.aliases = sorted(['production', 'qastaging', 'staging', 'dogfood', 'dev', 'test_dev', 'edge'])

    @contextmanager
    def edge_deprecation_error(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            yield
            self.assertEqual(len(caught), 1)
            warning, = caught
            self.assertTrue(issubclass(warning.category, DeprecationWarning))
            self.assertIn('no longer exists', str(warning))

    def test_short_names(self):
        self.assertEqual(sorted(uris.service_roots.keys()), self.aliases)
        self.assertEqual(sorted(uris.web_roots.keys()), self.aliases)

    def test_edge_service_root_is_production(self):
        with self.edge_deprecation_error():
            self.assertEqual(uris.lookup_service_root('edge'), uris.lookup_service_root('production'))

    def test_edge_web_root_is_production(self):
        with self.edge_deprecation_error():
            self.assertEqual(uris.lookup_web_root('edge'), uris.lookup_web_root('production'))

    def test_edge_service_root_url_becomes_production(self):
        with self.edge_deprecation_error():
            self.assertEqual(uris.lookup_service_root(uris.EDGE_SERVICE_ROOT), uris.lookup_service_root('production'))

    def test_edge_web_root_url_becomes_production(self):
        with self.edge_deprecation_error():
            self.assertEqual(uris.lookup_web_root(uris.EDGE_WEB_ROOT), uris.lookup_web_root('production'))

    def test_top_level_edge_constant_becomes_production(self):
        with self.edge_deprecation_error():
            self.assertEqual(uris.lookup_service_root(uris.EDGE_SERVICE_ROOT), uris.lookup_service_root('production'))

    def test_edge_server_equivalent_string_becomes_production(self):
        with self.edge_deprecation_error():
            self.assertEqual(uris.lookup_service_root('https://api.edge.launchpad.net/'), uris.lookup_service_root('production'))

    def test_edge_web_server_equivalent_string_becomes_production(self):
        with self.edge_deprecation_error():
            self.assertEqual(uris.lookup_web_root('https://edge.launchpad.net/'), uris.lookup_web_root('production'))

    def test_lookups(self):
        """Ensure that short service names turn into long service names."""
        with self.edge_deprecation_error():
            for alias in self.aliases:
                self.assertEqual(uris.lookup_service_root(alias), uris.service_roots[alias])
        with self.edge_deprecation_error():
            for alias in self.aliases:
                self.assertEqual(uris.lookup_web_root(alias), uris.web_roots[alias])
        other_root = 'http://some-other-server.com'
        self.assertEqual(uris.lookup_service_root(other_root), other_root)
        self.assertEqual(uris.lookup_web_root(other_root), other_root)
        not_a_url = 'not-a-url'
        self.assertRaises(ValueError, uris.lookup_service_root, not_a_url)
        self.assertRaises(ValueError, uris.lookup_web_root, not_a_url)