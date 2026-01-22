import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestDeriveToLocation(TestCase):
    """Test that the mapping of FROM_LOCATION to TO_LOCATION works."""

    def test_to_locations_derived_from_paths(self):
        derive = urlutils.derive_to_location
        self.assertEqual('bar', derive('bar'))
        self.assertEqual('bar', derive('../bar'))
        self.assertEqual('bar', derive('/foo/bar'))
        self.assertEqual('bar', derive('c:/foo/bar'))
        self.assertEqual('bar', derive('c:bar'))

    def test_to_locations_derived_from_urls(self):
        derive = urlutils.derive_to_location
        self.assertEqual('bar', derive('http://foo/bar'))
        self.assertEqual('bar', derive('bzr+ssh://foo/bar'))
        self.assertEqual('foo-bar', derive('lp:foo-bar'))