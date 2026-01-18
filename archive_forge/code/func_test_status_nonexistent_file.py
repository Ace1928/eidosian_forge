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
def test_status_nonexistent_file(self):
    wt = self._prepare_nonexistent()
    self.assertStatus(['removed:\n', '  FILE_E\n', 'added:\n', '  FILE_Q\n', 'modified:\n', '  FILE_B\n', '  FILE_C\n', 'unknown:\n', '  UNVERSIONED_BUT_EXISTING\n'], wt)
    self.assertStatus([' M  FILE_B\n', ' M  FILE_C\n', ' D  FILE_E\n', '+N  FILE_Q\n', '?   UNVERSIONED_BUT_EXISTING\n'], wt, short=True)
    expected = ['nonexistent:\n', '  NONEXISTENT\n']
    out, err = self.run_bzr('status NONEXISTENT', retcode=3)
    self.assertEqual(expected, out.splitlines(True))
    self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')
    expected = ['X:   NONEXISTENT\n']
    out, err = self.run_bzr('status --short NONEXISTENT', retcode=3)
    self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')