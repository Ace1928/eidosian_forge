import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_to_locations_derived_from_urls(self):
    derive = urlutils.derive_to_location
    self.assertEqual('bar', derive('http://foo/bar'))
    self.assertEqual('bar', derive('bzr+ssh://foo/bar'))
    self.assertEqual('foo-bar', derive('lp:foo-bar'))