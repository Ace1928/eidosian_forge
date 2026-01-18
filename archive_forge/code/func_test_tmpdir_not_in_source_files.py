import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def test_tmpdir_not_in_source_files(self):
    """When scanning for source files, we don't descend test tempdirs"""
    for filename in self.get_source_files():
        if re.search('test....\\.tmp', filename):
            self.fail('get_source_file() returned filename %r from within a temporary directory' % filename)