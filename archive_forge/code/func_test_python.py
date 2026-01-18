import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def test_python(self):
    """Test that we can do --format=python"""
    self.create_tree()
    txt = self.run_bzr('version-info --format python branch')[0]
    self.assertContainsRe(txt, 'version_info = {')