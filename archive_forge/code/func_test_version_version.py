import unittest
from traits.testing.optional_dependencies import (
import traits
@requires_pkg_resources
def test_version_version(self):
    from traits.version import version
    self.assertIsInstance(version, str)
    parsed_version = pkg_resources.parse_version(version)
    self.assertEqual(str(parsed_version), version)