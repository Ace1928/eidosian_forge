import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
def test_status_multiple_nonexistent_files(self):
    wt = self._prepare_nonexistent()
    expected = ['removed:\n', '  FILE_E\n', 'modified:\n', '  FILE_B\n', '  FILE_C\n', 'nonexistent:\n', '  ANOTHER_NONEXISTENT\n', '  NONEXISTENT\n']
    out, err = self.run_bzr('status NONEXISTENT FILE_A FILE_B ANOTHER_NONEXISTENT FILE_C FILE_D FILE_E', retcode=3)
    self.assertEqual(expected, out.splitlines(True))
    self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: ANOTHER_NONEXISTENT NONEXISTENT.*')
    expected = [' M  FILE_B\n', ' M  FILE_C\n', ' D  FILE_E\n', 'X   ANOTHER_NONEXISTENT\n', 'X   NONEXISTENT\n']
    out, err = self.run_bzr('status --short NONEXISTENT FILE_A FILE_B ANOTHER_NONEXISTENT FILE_C FILE_D FILE_E', retcode=3)
    self.assertEqual(expected, out.splitlines(True))
    self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: ANOTHER_NONEXISTENT NONEXISTENT.*')