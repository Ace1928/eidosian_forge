import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_to_locations_derived_from_paths(self):
    derive = urlutils.derive_to_location
    self.assertEqual('bar', derive('bar'))
    self.assertEqual('bar', derive('../bar'))
    self.assertEqual('bar', derive('/foo/bar'))
    self.assertEqual('bar', derive('c:/foo/bar'))
    self.assertEqual('bar', derive('c:bar'))