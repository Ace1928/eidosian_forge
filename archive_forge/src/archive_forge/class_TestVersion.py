import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
class TestVersion(TestCase):

    def test_main_version(self):
        """Check output from version command and master option is reasonable"""
        self.permit_source_tree_branch_repo()
        output = self.run_bzr('version')[0]
        self.log('brz version output:')
        self.log(output)
        self.assertTrue(output.startswith('Breezy (brz) '))
        self.assertNotEqual(output.index('Canonical'), -1)
        tmp_output = self.run_bzr('--version')[0]
        self.assertEqual(output, tmp_output)

    def test_version(self):
        self.permit_source_tree_branch_repo()
        out = self.run_bzr('version')[0]
        self.assertTrue(len(out) > 0)
        self.assertEqualDiff(out.splitlines()[0], 'Breezy (brz) %s' % breezy.__version__)
        self.assertContainsRe(out, '(?m)^  Python interpreter:')
        self.assertContainsRe(out, '(?m)^  Python standard library:')
        self.assertContainsRe(out, '(?m)^  breezy:')
        self.assertContainsRe(out, '(?m)^  Breezy configuration:')
        self.assertContainsRe(out, '(?m)^  Breezy log file:.*[\\\\/]breezy[\\\\/]brz\\.log')

    def test_version_short(self):
        self.permit_source_tree_branch_repo()
        out = self.run_bzr(['version', '--short'])[0]
        self.assertEqualDiff(out, breezy.version_string + '\n')