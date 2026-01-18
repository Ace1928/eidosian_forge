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
def test_status_nonexistent_file_with_unversioned(self):
    wt = self._prepare_nonexistent()
    expected = ['removed:\n', '  FILE_E\n', 'added:\n', '  FILE_Q\n', 'modified:\n', '  FILE_B\n', '  FILE_C\n', 'unknown:\n', '  UNVERSIONED_BUT_EXISTING\n', 'nonexistent:\n', '  NONEXISTENT\n']
    out, err = self.run_bzr('status NONEXISTENT FILE_A FILE_B UNVERSIONED_BUT_EXISTING FILE_C FILE_D FILE_E FILE_Q', retcode=3)
    self.assertEqual(expected, out.splitlines(True))
    self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')
    expected = sorted(['+N  FILE_Q\n', '?   UNVERSIONED_BUT_EXISTING\n', ' D  FILE_E\n', ' M  FILE_C\n', ' M  FILE_B\n', 'X   NONEXISTENT\n'])
    out, err = self.run_bzr('status --short NONEXISTENT FILE_A FILE_B UNVERSIONED_BUT_EXISTING FILE_C FILE_D FILE_E FILE_Q', retcode=3)
    actual = out.splitlines(True)
    actual.sort()
    self.assertEqual(expected, actual)
    self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')