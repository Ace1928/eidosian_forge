import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
class BisectHarnessTests(BisectTestCase):
    """Tests for the harness itself."""

    def testLastRev(self):
        """Test that the last revision is correct."""
        repo = self.tree.branch.repository
        top_revtree = repo.revision_tree(self.tree.last_revision())
        top_revtree.lock_read()
        top_file = top_revtree.get_file('test_file')
        test_content = top_file.read().strip()
        top_file.close()
        top_revtree.unlock()
        self.assertEqual(test_content, b'five')

    def testSubtreeRev(self):
        """Test that the last revision in a subtree is correct."""
        repo = self.tree.branch.repository
        sub_revtree = repo.revision_tree(self.subtree_rev)
        sub_revtree.lock_read()
        with sub_revtree.get_file('test_file') as sub_file:
            test_content = sub_file.read().strip()
        sub_revtree.unlock()
        self.assertEqual(test_content, b'one dot three')